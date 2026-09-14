"""
NEXUS Terminal UI
=================

Simple futuristic terminal interface for NEXUS.

This module only handles presentation.
It does not control the microphone, wake word,
Whisper, Ollama, or TTS.
"""

import os
import sys


# =========================================================
# ANSI / TERMINAL SUPPORT
# =========================================================

RESET = "\033[0m"
CYAN = "\033[96m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
WHITE = "\033[97m"
DIM = "\033[2m"


def enable_ansi():
    """
    Enable ANSI escape sequences on Windows terminals.
    """

    if os.name == "nt":
        os.system("")


def clear_screen():
    """
    Clear the terminal screen.
    """

    os.system("cls" if os.name == "nt" else "clear")


# =========================================================
# BANNER
# =========================================================

def show_banner():
    """
    Display the NEXUS startup banner.
    """

    enable_ansi()
    clear_screen()

    print(
        f"{CYAN}"
        "╔═══════════════════════════════════════════════════════╗\n"
        "║                                                       ║\n"
        "║                    N . E . X . U . S                  ║\n"
        "║                                                       ║\n"
        "║          Neural EXecution & Unified System            ║\n"
        "║                                                       ║\n"
        "╚═══════════════════════════════════════════════════════╝"
        f"{RESET}"
    )

    print()


# =========================================================
# STATUS
# =========================================================

def show_status(status, detail=""):
    """
    Display the current NEXUS system status.
    """

    colors = {
        "STANDBY": GREEN,
        "WAKE DETECTED": CYAN,
        "LISTENING": BLUE,
        "TRANSCRIBING": MAGENTA,
        "THINKING": YELLOW,
        "SPEAKING": CYAN,
        "ERROR": RED,
    }

    color = colors.get(status, WHITE)

    print()
    print(f"{DIM}───────────────────────────────────────────────────────{RESET}")
    print(f"  {color}NEXUS STATUS: {status}{RESET}")

    if detail:
        print(f"  {WHITE}{detail}{RESET}")

    print(f"{DIM}───────────────────────────────────────────────────────{RESET}")
    print()


# =========================================================
# MESSAGES
# =========================================================

def show_user_message(message):
    """
    Display something RIO said.
    """

    print(f"{GREEN}RIO:{RESET} {message}")


def show_nexus_message(message):
    """
    Display something NEXUS said.
    """

    print(f"{CYAN}NEXUS:{RESET} {message}")


def show_error(message):
    """
    Display an error.
    """

    print(f"{RED}NEXUS ERROR:{RESET} {message}")


# =========================================================
# STANDBY
# =========================================================

def show_standby():
    """
    Display the normal waiting state.
    """

    show_status(
        "STANDBY",
        "Listening for: Hey NEXUS"
    )


# =========================================================
# READY
# =========================================================

def show_ready():
    """
    Display that NEXUS is ready.
    """

    show_status(
        "STANDBY",
        "NEXUS is online • Ready for RIO"
    )