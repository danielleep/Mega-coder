"""Mega Coder console menu."""


MENU = """I’m Mega Coder. What would you like me to do today?

1. Develop a python program.
2. Fix/change something in a Github repository.
3. Look at my screen and give me realtime coding tips."""


def show_menu():
    """Display the Mega Coder menu."""
    print(MENU)


def ask_for_program_description():
    """Ask for and return a non-empty program description."""
    print("Describe me which python program you want me to develop")

    while True:
        description = input().strip()
        if description:
            return description
        print("The program description cannot be empty. Please try again.")


def main():
    """Run the Mega Coder menu."""
    try:
        while True:
            show_menu()
            choice = input().strip()

            if choice == "1":
                program_description = ask_for_program_description()
                return

            if choice in {"2", "3"}:
                print("Not implemented yet")
            else:
                print("Please choose 1, 2, or 3.")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
