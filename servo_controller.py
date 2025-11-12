#!/usr/bin/env python3
import time
try:
    import lgpio
except Exception:
    lgpio = None

class ServoController:
    def __init__(self, pin=18, frequency=50, min_us=500, max_us=2500, cooldown=3.0, debug=False):
        self.pin = pin
        self.freq = frequency
        self.min_us = min_us
        self.max_us = max_us
        self.debug = debug
        self.current_angle = 90.0
        self._dry = False
        self._last_fire = 0.0
        self._cooldown = float(cooldown)
        self.lgpio = None
        self.chip = None
        if lgpio is None:
            self._dry = True
            if self.debug:
                print("[servo_controller] lgpio not available, running in dry mode")
        else:
            try:
                self.lgpio = lgpio
                self.chip = lgpio.gpiochip_open(0)
                lgpio.gpio_claim_output(self.chip, self.pin)
                self.set_angle(self.current_angle, hold=0.1)
                if self.debug:
                    print("[servo_controller] lgpio initialized")
            except Exception as e:
                self._dry = True
                self.lgpio = None
                self.chip = None
                if self.debug:
                    print(f"[servo_controller] failed to init lgpio, dry mode: {e}")

    def _angle_to_duty(self, angle):
        angle = max(0.0, min(180.0, float(angle)))
        pulse_range = self.max_us - self.min_us
        pulse = self.min_us + (angle / 180.0) * pulse_range
        period = 1_000_000.0 / self.freq
        duty = (pulse / period) * 100.0
        if self.debug:
            print(f"[servo_controller debug] angle={angle:.1f} -> {pulse:.0f}us -> duty={duty:.2f}%")
        return duty

    def set_angle(self, angle, hold=0.1):
        angle = max(0.0, min(180.0, float(angle)))
        self.current_angle = angle
        if self._dry:
            if self.debug:
                print(f"[servo_controller dry] set_angle({self.current_angle:.1f}) hold={hold}")
            time.sleep(hold)
            return
        try:
            duty = self._angle_to_duty(self.current_angle)
            res = self.lgpio.tx_pwm(self.chip, self.pin, self.freq, duty)
            if res < 0 and self.debug:
                print(f"[servo_controller] tx_pwm error: {res}")
            time.sleep(hold)
        except Exception as e:
            if self.debug:
                print(f"[servo_controller] set_angle error: {e}")

    def fire(self, angle=30.0, rest=90.0, hold=0.12, back=0.08):
        now = time.time()
        if now - self._last_fire < self._cooldown:
            if self.debug:
                print("[servo_controller] fire requested but still cooling down")
            return False
        self._last_fire = now
        if self.debug:
            print("[servo_controller] firing")
        self.set_angle(angle, hold=hold)
        self.set_angle(rest, hold=back)
        return True

    def center(self):
        self.set_angle(90.0, hold=0.1)

    def off(self):
        if self._dry:
            return
        try:
            self.lgpio.tx_pwm(self.chip, 0, 0, 0)
        except Exception:
            pass

    def cleanup(self):
        if self._dry:
            return
        try:
            self.lgpio.tx_pwm(self.chip, 0, 0, 0)
            self.lgpio.gpiochip_close(self.chip)
        except Exception:
            pass
