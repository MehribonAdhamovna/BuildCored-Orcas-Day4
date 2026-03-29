# BuildCored-Orcas-Day4
BlinkLock — BUILDCORED ORCAS Day 04

What it does. 
The system uses computer vision to monitor the Eye Aspect Ratio, locking the screen after three rapid blinks and unlocking only after a deliberate, long-hold wink.

Hardware concept. 
This project mimics a digital state machine with software-based debouncing, treating eye blinks like physical buttons that require filtered, timed signals to prevent accidental triggers.

What I would do differently. 
I would implement an adaptive threshold that automatically calibrates to the user's unique eye shape and ambient lighting to reduce false positives during normal activity.

Run it. python day04_starter.py
