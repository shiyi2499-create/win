"""
Keystroke Vibration Data Collector - Main Orchestrator
======================================================
Coordinates sensor reading, keyboard listening, and rate monitoring.
Supports two modes:
  1. single_key  - guided per-key repeated pressing
  2. free_type   - natural typing with auto key logging

FIXED v2:
  - Background drain thread runs continuously in ALL modes
    (no more rate drops during pauses between keys)
  - Rate monitor uses simple counter-based calculation
  - Sensor reader handles None samples gracefully

Run with:  sudo python3 collector.py --mode single_key
           sudo python3 collector.py --mode free_type
"""

import os
import csv
import sys
import time
import argparse
import threading
from datetime import datetime
from typing import Optional

from config import CollectorConfig
from sensor_reader import SensorReader, SensorSample
from keyboard_listener import KeyboardListener, KeyEvent
from rate_monitor import RateMonitor


class DataCollector:
    """
    Main data collection coordinator.
    """

    def __init__(self, config: CollectorConfig, mode: str, group: int = 0,
                 free_type_part: int = 0):
        self.cfg = config
        self.mode = mode
        self.group = group
        self._free_type_part = free_type_part

        # Components
        self.sensor = SensorReader()
        self.keyboard = KeyboardListener()
        self.rate_monitor = RateMonitor(
            min_rate_hz=config.MIN_ACCEPTABLE_RATE_HZ,
            check_interval_sec=config.RATE_CHECK_INTERVAL_SEC,
            on_rate_drop=self._on_rate_drop,
        )

        # Session info
        self.session_prefix = config.session_prefix(mode, group, free_type_part)
        self.sensor_csv_path = os.path.join(
            config.RAW_DIR, f"{self.session_prefix}_sensor.csv"
        )
        self.events_csv_path = os.path.join(
            config.RAW_DIR, f"{self.session_prefix}_events.csv"
        )
        self.meta_path = os.path.join(
            config.RAW_DIR, f"{self.session_prefix}_meta.txt"
        )

        # State
        self._stop_event = threading.Event()
        self._rate_drop_detected = False
        self._session_valid = True

        # Counters
        self._sensor_count = 0
        self._event_count = 0

        # CSV writers
        self._sensor_file = None
        self._sensor_writer = None
        self._events_file = None
        self._events_writer = None
        self._csv_lock = threading.Lock()

        # For single_key mode: track per-key press count from drain thread
        self._current_target_key: Optional[str] = None
        self._target_press_count = 0
        self._target_press_lock = threading.Lock()

    # ── Rate drop callback ───────────────────────────────────

    def _on_rate_drop(self, rate: float):
        self._rate_drop_detected = True
        self._stop_event.set()
        print(
            f"\n{'='*60}\n"
            f"  ⚠️  SAMPLING RATE ALERT!\n"
            f"  Rate dropped to {rate:.1f} Hz (minimum: {self.cfg.MIN_ACCEPTABLE_RATE_HZ} Hz)\n"
            f"  Recording stopped. Data collected so far has been SAVED.\n"
            f"{'='*60}"
        )

    # ── CSV I/O ──────────────────────────────────────────────

    def _open_csv_files(self):
        self._sensor_file = open(self.sensor_csv_path, "w", newline="")
        self._sensor_writer = csv.writer(self._sensor_file)
        self._sensor_writer.writerow([
            "timestamp_ns", "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z"
        ])

        self._events_file = open(self.events_csv_path, "w", newline="")
        self._events_writer = csv.writer(self._events_file)
        self._events_writer.writerow([
            "timestamp_ns", "key", "event_type", "participant_id", "session_id"
        ])

    def _close_csv_files(self):
        if self._sensor_file:
            self._sensor_file.flush()
            self._sensor_file.close()
        if self._events_file:
            self._events_file.flush()
            self._events_file.close()

    def _write_sensor_samples(self, samples: list[SensorSample]):
        for s in samples:
            self._sensor_writer.writerow([
                s.timestamp_ns,
                f"{s.accel_x:.8f}", f"{s.accel_y:.8f}", f"{s.accel_z:.8f}",
                f"{s.gyro_x:.6f}", f"{s.gyro_y:.6f}", f"{s.gyro_z:.6f}",
            ])
        self._sensor_count += len(samples)

    def _write_key_events(self, events: list[KeyEvent]):
        for e in events:
            self._events_writer.writerow([
                e.timestamp_ns, e.key, e.event_type,
                self.cfg.PARTICIPANT_ID, self.session_prefix,
            ])
        self._event_count += len(events)

    # ── Background drain thread (runs in ALL modes) ──────────

    def _drain_thread_fn(self):
        """
        Continuously drains sensor + keyboard buffers to CSV.
        This runs the ENTIRE session, including pauses between keys.
        This ensures the rate monitor always gets ticks.
        """
        flush_counter = 0

        while not self._stop_event.is_set():
            # Drain sensor
            samples = self.sensor.drain()
            if samples:
                with self._csv_lock:
                    self._write_sensor_samples(samples)
                self.rate_monitor.tick(count=len(samples))

            # Drain keyboard events
            events = self.keyboard.drain()
            if events:
                with self._csv_lock:
                    self._write_key_events(events)

                # Count target key presses for single_key mode
                if self._current_target_key is not None:
                    with self._target_press_lock:
                        for e in events:
                            if (e.key == self._current_target_key
                                    and e.event_type == "press"):
                                self._target_press_count += 1

            # Periodic flush
            flush_counter += 1
            if flush_counter >= 20:  # every ~2s
                with self._csv_lock:
                    if self._sensor_file:
                        self._sensor_file.flush()
                    if self._events_file:
                        self._events_file.flush()
                flush_counter = 0

            time.sleep(0.1)

        # Final drain
        samples = self.sensor.drain()
        if samples:
            with self._csv_lock:
                self._write_sensor_samples(samples)
        events = self.keyboard.drain()
        if events:
            with self._csv_lock:
                self._write_key_events(events)

    # ── Single Key Mode ──────────────────────────────────────

    def _run_single_key_mode(self):
        group_info = ""
        if self.group > 0:
            g = self.cfg.KEY_GROUPS[self.group]
            group_info = f"  Group:     {g['name']}\n"

        print(
            f"\n{'='*60}\n"
            f"  SINGLE KEY MODE\n"
            f"{group_info}"
            f"  Keys:      {' '.join(self.cfg.KEY_LIST)}\n"
            f"  Repeats:   {self.cfg.REPEATS_PER_KEY} per key\n"
            f"  Total:     {len(self.cfg.KEY_LIST)} keys × "
            f"{self.cfg.REPEATS_PER_KEY} = "
            f"{len(self.cfg.KEY_LIST) * self.cfg.REPEATS_PER_KEY} presses\n"
            f"  Ctrl+C to stop early.\n"
            f"{'='*60}\n"
        )

        print("  Warming up sensor (3 seconds)...")
        time.sleep(3)
        if self._stop_event.is_set():
            return
        print("  Sensor ready! Let's go.\n")

        for idx, target_key in enumerate(self.cfg.KEY_LIST):
            if self._stop_event.is_set():
                break

            remaining = len(self.cfg.KEY_LIST) - idx
            print(
                f"  [{idx+1}/{len(self.cfg.KEY_LIST)}] "
                f"Press  [ {target_key} ]  × {self.cfg.REPEATS_PER_KEY}  "
                f"(remaining: {remaining})"
            )

            # Reset target key press counter
            with self._target_press_lock:
                self._current_target_key = target_key
                self._target_press_count = 0

            # Wait for user to finish pressing
            last_display = -1
            while not self._stop_event.is_set():
                with self._target_press_lock:
                    count = self._target_press_count

                if count >= self.cfg.REPEATS_PER_KEY:
                    break

                # Update progress bar
                if count != last_display:
                    bar_len = 30
                    filled = int(bar_len * count / self.cfg.REPEATS_PER_KEY)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    rate = self.rate_monitor.current_rate
                    print(
                        f"\r    [{bar}] {count}/{self.cfg.REPEATS_PER_KEY}  "
                        f"Rate: {rate:.0f}Hz  ",
                        end="", flush=True
                    )
                    last_display = count

                time.sleep(0.05)

            # Final bar
            rate = self.rate_monitor.current_rate
            bar = "█" * 30
            with self._target_press_lock:
                final_count = self._target_press_count
            print(
                f"\r    [{bar}] {final_count}/{self.cfg.REPEATS_PER_KEY}  "
                f"Rate: {rate:.0f}Hz  "
            )
            print(f"    ✓ '{target_key}' done!")

            # Clear target so drain thread stops counting for old key
            with self._target_press_lock:
                self._current_target_key = None

            # Pause between keys (drain thread keeps running!)
            if idx < len(self.cfg.KEY_LIST) - 1 and not self._stop_event.is_set():
                print(f"    (pause {self.cfg.PAUSE_BETWEEN_KEYS_SEC}s...)")
                time.sleep(self.cfg.PAUSE_BETWEEN_KEYS_SEC)

        print("\n  All keys completed!")

    # ── Free Type Mode (Guided with validation) ────────────────

    def _run_free_type_mode(self):
        from typing_prompts import PROMPTS

        # Split into 3 parts
        part_size = (len(PROMPTS) + 2) // 3   # ceiling division
        parts = {
            1: PROMPTS[0:part_size],
            2: PROMPTS[part_size:2*part_size],
            3: PROMPTS[2*part_size:],
        }

        part = self._free_type_part
        if part not in parts:
            prompts = PROMPTS
            part_label = "ALL"
        else:
            prompts = parts[part]
            part_label = f"Part {part}/3"

        print(
            f"\n{'='*60}\n"
            f"  FREE TYPE MODE (Guided)\n"
            f"{'='*60}\n"
            f"  {part_label}: {len(prompts)} sentences\n"
            f"\n"
            f"  How it works:\n"
            f"    1. A sentence appears → type it exactly\n"
            f"    2. Press ENTER to submit\n"
            f"    3. If correct → next sentence\n"
            f"       If wrong  → shows the diff, you retype\n"
            f"\n"
            f"  Tips:\n"
            f"    - All lowercase, no need for shift\n"
            f"    - Backspace to fix typos before pressing ENTER\n"
            f"    - Ctrl+C to stop early\n"
            f"{'='*60}\n"
        )

        print("  Warming up sensor (3 seconds)...")
        time.sleep(3)
        if self._stop_event.is_set():
            return
        print("  Sensor ready! Start typing.\n")

        # Prompts log
        prompts_log_path = self.events_csv_path.replace("_events.csv", "_prompts.csv")
        prompts_file = open(prompts_log_path, "w", newline="")
        prompts_writer = csv.writer(prompts_file)
        prompts_writer.writerow(["prompt_index", "timestamp_ns", "prompt_text", "typed_text", "match"])

        completed = 0
        for idx, prompt in enumerate(prompts):
            if self._stop_event.is_set():
                break

            matched = False
            attempt = 0

            while not matched and not self._stop_event.is_set():
                attempt += 1

                # Display prompt
                if attempt == 1:
                    print(f"  ┌─ Sentence {idx+1}/{len(prompts)} ──────────────────")
                else:
                    print(f"  ┌─ Sentence {idx+1}/{len(prompts)} (retry #{attempt}) ───")
                print(f"  │  {prompt}")
                print(f"  └─────────────────────────────────────────")

                prompt_ts = time.perf_counter_ns()

                # Track typed characters in a buffer
                typed_buffer = []

                waiting = True
                while waiting and not self._stop_event.is_set():
                    events = self.keyboard.drain()
                    if events:
                        with self._csv_lock:
                            self._write_key_events(events)

                        for e in events:
                            if e.event_type != "press":
                                continue

                            if e.key == "enter":
                                waiting = False
                                break
                            elif e.key == "backspace":
                                if typed_buffer:
                                    typed_buffer.pop()
                            elif e.key == "space":
                                typed_buffer.append(" ")
                            elif len(e.key) == 1:
                                # Single character (a-z, 0-9, punctuation)
                                typed_buffer.append(e.key)
                            # Ignore other special keys (shift, ctrl, etc.)

                    # Sensor draining is handled by the background drain thread.
                    # Only sleep here to avoid busy-spinning on keyboard events.
                    time.sleep(0.05)

                # Check what was typed
                typed_text = "".join(typed_buffer).strip()
                expected = prompt.strip()

                if typed_text == expected:
                    matched = True
                    rate = self.rate_monitor.current_rate
                    prompts_writer.writerow([idx, prompt_ts, prompt, typed_text, "YES"])
                    prompts_file.flush()
                    completed += 1
                    print(f"    ✅ Correct! ({completed}/{len(prompts)})  "
                          f"Rate: {rate:.0f}Hz\n")
                else:
                    # Show diff
                    prompts_writer.writerow([idx, prompt_ts, prompt, typed_text, "NO"])
                    prompts_file.flush()
                    print(f"    ❌ Mismatch! Please retype.")
                    print(f"    Expected: {expected}")
                    print(f"    You typed: {typed_text}")
                    # Find first difference position
                    for di in range(min(len(expected), len(typed_text))):
                        if di >= len(typed_text) or expected[di] != typed_text[di]:
                            print(f"    Diff at position {di}: "
                                  f"expected '{expected[di] if di < len(expected) else '(end)'}' "
                                  f"got '{typed_text[di] if di < len(typed_text) else '(end)'}'")
                            break
                    else:
                        if len(typed_text) != len(expected):
                            print(f"    Length: expected {len(expected)}, got {len(typed_text)}")
                    print()

        prompts_file.close()
        print(f"\n  Done! {completed} sentences completed.")
        print(f"  Prompts log: {prompts_log_path}")

    # ── Metadata ─────────────────────────────────────────────

    def _save_metadata(self, start_time: float, end_time: float):
        rate_summary = self.rate_monitor.get_rate_summary()
        with open(self.meta_path, "w") as f:
            f.write(f"Session: {self.session_prefix}\n")
            f.write(f"Mode: {self.mode}\n")
            f.write(f"Round: {self.cfg.ROUND}\n")
            if self.group > 0:
                g = self.cfg.KEY_GROUPS[self.group]
                f.write(f"Group: {self.group} - {g['name']}\n")
                f.write(f"Keys in group: {' '.join(g['keys'])}\n")
            if self._free_type_part > 0:
                f.write(f"Free type part: {self._free_type_part}/3\n")
            f.write(f"Participant: {self.cfg.PARTICIPANT_ID}\n")
            f.write(f"Start: {datetime.fromtimestamp(start_time).isoformat()}\n")
            f.write(f"End: {datetime.fromtimestamp(end_time).isoformat()}\n")
            f.write(f"Duration: {end_time - start_time:.1f}s\n")
            f.write(f"Total sensor samples: {self._sensor_count}\n")
            f.write(f"Total key events: {self._event_count}\n")
            f.write(f"Rate drop detected: {self._rate_drop_detected}\n")
            f.write(f"Session valid: {self._session_valid}\n")
            f.write(f"Sampling rate - min: {rate_summary['min']:.1f} Hz\n")
            f.write(f"Sampling rate - max: {rate_summary['max']:.1f} Hz\n")
            f.write(f"Sampling rate - avg: {rate_summary['avg']:.1f} Hz\n")
            f.write(f"Sensor CSV: {self.sensor_csv_path}\n")
            f.write(f"Events CSV: {self.events_csv_path}\n")

    # ── Public run ───────────────────────────────────────────

    def run(self):
        start_wall = time.time()

        print(f"\n  Session: {self.session_prefix}")
        print(f"  Sensor CSV: {self.sensor_csv_path}")
        print(f"  Events CSV: {self.events_csv_path}")

        self._open_csv_files()

        # Start sensor
        try:
            self.sensor.start()
        except Exception as e:
            print(f"\n  ❌ Failed to start sensor: {e}")
            print("  Make sure you're running with sudo on Apple Silicon Mac.")
            self._close_csv_files()
            return

        # Start keyboard listener & rate monitor
        self.keyboard.start()
        self.rate_monitor.start()

        # Start background drain thread (runs in ALL modes).
        # Previously free_type handled draining inside its own prompt loop,
        # which caused rate monitor dead zones during sentence-reading pauses.
        # Now the drain thread always runs; free_type's inner loop still drains
        # keyboard events for prompt validation, but sensor+rate go through here.
        drain_thread = threading.Thread(target=self._drain_thread_fn, daemon=True)
        drain_thread.start()

        try:
            if self.mode == "single_key":
                self._run_single_key_mode()
            elif self.mode == "free_type":
                self._run_free_type_mode()
            else:
                print(f"  Unknown mode: {self.mode}")

        except KeyboardInterrupt:
            print("\n\n  ⏹  Stopped by user (Ctrl+C)")

        finally:
            end_wall = time.time()

            # Signal everything to stop
            self._stop_event.set()

            # Wait for drain thread to finish final flush
            drain_thread.join(timeout=3.0)

            # Stop components
            self.rate_monitor.stop()
            self.keyboard.stop()
            self.sensor.stop()
            self._close_csv_files()

            self._session_valid = not self._rate_drop_detected
            self._save_metadata(start_wall, end_wall)

            rate_summary = self.rate_monitor.get_rate_summary()
            duration = end_wall - start_wall

            print(
                f"\n{'='*60}\n"
                f"  SESSION SUMMARY\n"
                f"{'='*60}\n"
                f"  Duration:        {duration:.1f}s\n"
                f"  Sensor samples:  {self._sensor_count:,}\n"
                f"  Key events:      {self._event_count:,}\n"
                f"  Avg rate:        {rate_summary['avg']:.1f} Hz\n"
                f"  Min rate:        {rate_summary['min']:.1f} Hz\n"
                f"  Max rate:        {rate_summary['max']:.1f} Hz\n"
                f"  Rate drop:       {'YES ⚠️' if self._rate_drop_detected else 'No ✓'}\n"
                f"  Session valid:   {'YES ✓' if self._session_valid else 'NO ⚠️'}\n"
                f"{'='*60}\n"
                f"  Files saved:\n"
                f"    {self.sensor_csv_path}\n"
                f"    {self.events_csv_path}\n"
                f"    {self.meta_path}\n"
                f"{'='*60}\n"
            )

            if not self._session_valid:
                print(
                    "  ⚠️  Rate drop detected. Data before the drop is still\n"
                    "  valid and saved. Consider re-running this session.\n"
                )


# ── CLI ──────────────────────────────────────────────────────

def show_group_menu(cfg: CollectorConfig) -> int:
    """Interactive group selection menu. Returns group number 1-7 or 0 for all."""
    print(
        f"\n  ┌──────────────────────────────────────────────┐\n"
        f"  │           SELECT A KEY GROUP                  │\n"
        f"  ├──────────────────────────────────────────────┤"
    )
    for gid, g in cfg.KEY_GROUPS.items():
        keys_str = "  ".join(g["keys"])
        print(f"  │  {gid})  {g['name']:<22s}  [ {keys_str} ]")
    print(
        f"  │  0)  ALL groups at once                      │\n"
        f"  └──────────────────────────────────────────────┘"
    )

    max_group = max(cfg.KEY_GROUPS.keys())
    while True:
        try:
            choice = input(f"\n  Enter group number (0-{max_group}): ").strip()
            choice = int(choice)
            if 0 <= choice <= max_group:
                return choice
            print(f"  Invalid choice, enter 0-{max_group}.")
        except (ValueError, EOFError):
            print(f"  Invalid input, enter a number 0-{max_group}.")


def main():
    parser = argparse.ArgumentParser(
        description="Keystroke Vibration Data Collector"
    )
    parser.add_argument(
        "--mode", choices=["single_key", "free_type"],
        default="single_key",
        help="Collection mode (default: single_key)"
    )
    parser.add_argument(
        "--participant", default="p01",
        help="Participant ID (default: p01)"
    )
    parser.add_argument(
        "--repeats", type=int, default=50,
        help="Presses per key in single_key mode (default: 50)"
    )
    parser.add_argument(
        "--min-rate", type=int, default=96,
        help="Minimum acceptable sampling rate in Hz (default: 96)"
    )
    parser.add_argument(
        "--group", type=int, default=-1,
        help="Key group 1-6, or 0 for all (default: interactive menu)"
    )
    parser.add_argument(
        "--round", type=int, default=1,
        help="Data collection round number (default: 1). "
             "Data saves to data/raw/round{N}/"
    )
    parser.add_argument(
        "--part", type=int, default=0,
        help="Free type part: 1, 2, or 3 (splits 39 sentences into ~13 each). "
             "0 = all sentences (default: 0)"
    )
    args = parser.parse_args()

    cfg = CollectorConfig(
        PARTICIPANT_ID=args.participant,
        REPEATS_PER_KEY=args.repeats,
        MIN_ACCEPTABLE_RATE_HZ=args.min_rate,
        ROUND=args.round,
    )

    print(
        f"\n{'='*60}\n"
        f"  🎹 KEYSTROKE VIBRATION DATA COLLECTOR\n"
        f"{'='*60}\n"
        f"  Mode:        {args.mode}\n"
        f"  Round:       {args.round}  (→ {cfg.RAW_DIR})\n"
        f"  Participant: {args.participant}\n"
        f"  Min rate:    {args.min_rate} Hz\n"
    )

    # Group selection (only for single_key mode)
    group = 0
    if args.mode == "single_key":
        if args.group == -1:
            # Interactive menu
            group = show_group_menu(cfg)
        else:
            group = args.group

        if group == 0:
            # All keys
            cfg.KEY_LIST = []
            for g in cfg.KEY_GROUPS.values():
                cfg.KEY_LIST.extend(g["keys"])
            print(f"  Selected:    ALL keys ({len(cfg.KEY_LIST)} keys)")
        else:
            g = cfg.KEY_GROUPS[group]
            cfg.KEY_LIST = g["keys"]
            print(f"  Selected:    {g['name']}")
            print(f"  Keys:        {' '.join(g['keys'])}")

        print(f"  Repeats/key: {args.repeats}")
        total = len(cfg.KEY_LIST) * args.repeats
        print(f"  Total:       {len(cfg.KEY_LIST)} keys × {args.repeats} = {total} presses\n")

    if args.mode == "free_type" and args.part > 0:
        print(f"  Part:        {args.part}/3\n")

    if os.geteuid() != 0:
        print("  ⚠️  Not running as root! SPU sensor requires sudo.")
        print("  Run:  sudo .venv/bin/python3 collector.py --mode single_key\n")
        sys.exit(1)

    collector = DataCollector(cfg, args.mode, group, free_type_part=args.part)
    collector.run()


if __name__ == "__main__":
    main()