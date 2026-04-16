from rich.console import Console

console = Console()

class DataLogger:
    @staticmethod
    def info(message: str):
        console.print(f"[bold blue][INFO][/bold blue] {message}")

    @staticmethod
    def action(message: str):
        console.print(f"[bold yellow][ACTION][/bold yellow] {message}")

    @staticmethod
    def success(message: str):
        console.print(f"[bold green][SUCCESS][/bold green] {message}")

    @staticmethod
    def error(message: str):
        console.print(f"[bold red][ERROR][/bold red] {message}")

    @staticmethod
    def warn(message: str):
        console.print(f"[bold magenta][WARN][/bold magenta] {message}")
