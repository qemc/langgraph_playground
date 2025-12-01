"""
Rich Terminal UI for LangGraph Quiz Game
A beautiful, single-window terminal interface with no scrolling.
"""

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.align import Align
from langchain_core.messages import HumanMessage
from typing import Optional
import os


class RichQuizUI:
    """Manages the Rich terminal UI for the quiz game."""
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        
        # Game state tracking
        self.score = 0
        self.hit_points = 0
        self.topic = ""
        self.tokens_so_far = 0
        self.current_question = ""
        self.hints = []
        self.feedback_result = None
        self.feedback_justification = ""
        
        # Setup layout structure
        self._setup_layout()
    
    def _setup_layout(self):
        """Create the fixed layout structure."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="stats", size=4),
            Layout(name="question", size=8),
            Layout(name="hints", size=4),
            Layout(name="feedback", size=5)
        )
    
    def _render_header(self) -> Panel:
        """Render the header panel."""
        title = Text("🎯 QUIZ MASTER 🎯", style="bold magenta", justify="center")
        return Panel(title, style="bold blue", padding=0)
    
    def _render_stats(self) -> Panel:
        """Render the stats panel with score, HP, topic, and tokens."""
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="cyan", width=12)
        table.add_column(justify="left", style="green")
        
        # Score with progress visualization
        score_bar = "█" * self.score + "░" * (3 - self.score)
        
        # Hit Points with color coding
        hp_color = "green" if self.hit_points > 1 else "red" if self.hit_points == 1 else "dim red"
        hp_hearts = "❤️ " * self.hit_points + "🖤 " * (3 - self.hit_points)
        
        # Compact display
        table.add_row("Score:", f"{self.score}/3 {score_bar}  |  HP: {self.hit_points}/3 {hp_hearts}")
        table.add_row("Topic:", f"{self.topic}  |  Tokens: {int(self.tokens_so_far)}")
        
        return Panel(table, title="📊 Stats", style="cyan", border_style="cyan", padding=0)
    
    def _render_question(self) -> Panel:
        """Render the current question panel."""
        if not self.current_question:
            content = Text("Waiting for question...", style="dim italic")
        else:
            # Truncate if too long (max ~400 chars to fit panel)
            question_text = self.current_question[:400] + "..." if len(self.current_question) > 400 else self.current_question
            content = Text(question_text, style="white")
        
        return Panel(
            Align.left(content),
            title="❓ Current Question",
            style="yellow",
            border_style="yellow",
            padding=(0, 1)
        )
    
    def _render_hints(self) -> Panel:
        """Render hints panel showing recent hints."""
        if not self.hints:
            content = Text("No hints requested", style="dim italic")
        else:
            # Show only the last hint, truncated
            last_hint = self.hints[-1]
            hint_text = last_hint[:150] + "..." if len(last_hint) > 150 else last_hint
            content = Text(f"💡 {hint_text}", style="yellow")
        
        return Panel(
            Align.left(content),
            title="💡 Latest Hint",
            style="yellow",
            border_style="yellow",
            padding=(0, 1)
        )
    
    def _render_feedback(self) -> Panel:
        """Render feedback panel with judge's result."""
        if self.feedback_result is None:
            content = Text("Waiting for your answer...", style="dim italic")
            style = "white"
        else:
            if self.feedback_result:
                result_text = Text("✅ CORRECT!", style="bold green")
                style = "green"
            else:
                result_text = Text("❌ WRONG!", style="bold red")
                style = "red"
            
            # Truncate long justifications
            just_text = self.feedback_justification[:200] + "..." if len(self.feedback_justification) > 200 else self.feedback_justification
            justification = Text(f"\n{just_text}", style="white")
            content = result_text + justification
        
        return Panel(
            Align.left(content),
            title="📝 Judge Feedback",
            style=style,
            border_style=style,
            padding=(0, 1)
        )
    
    def update_display(self):
        """Update all panels in the layout."""
        self.layout["header"].update(self._render_header())
        self.layout["stats"].update(self._render_stats())
        self.layout["question"].update(self._render_question())
        self.layout["hints"].update(self._render_hints())
        self.layout["feedback"].update(self._render_feedback())
    
    def update_stats(self, score: int, hit_points: int, topic: str, tokens: float):
        """Update game statistics."""
        self.score = score
        self.hit_points = hit_points
        self.topic = topic
        self.tokens_so_far = tokens
    
    def show_question(self, question: str):
        """Display a new question."""
        self.current_question = question
        self.feedback_result = None
        self.feedback_justification = ""
    
    def add_hint(self, hint: str):
        """Add a hint to the hints list."""
        self.hints.append(hint)
    
    def show_feedback(self, result: bool, justification: str):
        """Show judge feedback."""
        self.feedback_result = result
        self.feedback_justification = justification
    
    def clear_feedback(self):
        """Clear feedback for next question."""
        self.feedback_result = None
        self.feedback_justification = ""


def run_quiz_with_rich_ui():
    """Run the quiz game with Rich UI - replacement for run_hitl()."""
    import main
    from main import app, initial_state
    import os
    
    # Enable Rich UI mode to suppress all print statements
    main.RICH_UI_MODE = True
    
    ui = RichQuizUI()
    config = {'configurable': {"thread_id": "session_v1"}}
    
    # Initialize display
    ui.update_stats(
        initial_state['score'],
        initial_state['hitPoints'],
        initial_state['topic'],
        initial_state['tokens_so_far']
    )
    
    # Initial invoke
    app.invoke(initial_state, config=config)
    
    while True:
        snapshot = app.get_state(config)
        
        if not snapshot.next:
            ui.console.clear()
            ui.update_display()
            ui.console.print(ui.layout)
            ui.console.print("[cyan]Press ENTER to exit...[/cyan]", end="")
            input()
            break
        
        next_node = snapshot.next[0]
        state = snapshot.values
        
        # Update UI with current state
        ui.update_stats(
            state.get('score', 0),
            state.get('hitPoints', 0),
            state.get('topic', ''),
            state.get('tokens_so_far', 0)
        )
        
        # Show current question if available
        if state.get('questions'):
            last_question = state['questions'][-1].content
            ui.show_question(last_question)
        
        # Show hints if available
        if state.get('hints'):
            for hint_dict in state['hints']:
                hint_text = hint_dict.get('hint', '')
                if hint_text and hint_text not in ui.hints:
                    ui.add_hint(hint_text)
        
        # Show assessment feedback if available
        if state.get('assessments') and len(state['assessments']) > 0:
            last_assessment = state['assessments'][-1]
            if hasattr(last_assessment, 'tool_calls') and last_assessment.tool_calls:
                tool_call = last_assessment.tool_calls[0]
                if tool_call['name'] == 'JudgeAnswer':
                    args = tool_call['args']
                    ui.show_feedback(args.get('result', False), args.get('just', ''))
        
        if next_node == 'human_answer':
            # Clear screen and show UI
            ui.console.clear()
            ui.update_display()
            ui.console.print(ui.layout)
            
            # Get user input
            user_input = ui.console.input("[cyan]⌨️  Type your answer: [/cyan]")
            
            if user_input.lower() in ['q', 'quit']:
                break
            
            app.update_state(config, {'user_answers': [HumanMessage(content=user_input)]})
            
            # Show thinking message
            ui.console.clear()
            ui.update_display()
            ui.console.print(ui.layout)
            ui.console.print("[yellow]⏳ Thinking...[/yellow]")
            
            app.invoke(None, config=config)
        
        elif next_node == 'human_next':
            # Check if game ended (won or lost)
            current_score = state.get('score', 0)
            current_hp = state.get('hitPoints', 0)
            
            if current_hp == 0:
                ui.show_feedback(False, "You lost the game! 💀")
                ui.console.clear()
                ui.update_display()
                ui.console.print(ui.layout)
                ui.console.print("[red]Press ENTER to exit...[/red]", end="")
                input()
                break
            elif current_score >= 3:
                ui.show_feedback(True, "You won the game! 🎉")
                ui.console.clear()
                ui.update_display()
                ui.console.print(ui.layout)
                ui.console.print("[green]Press ENTER to exit...[/green]", end="")
                input()
                break
            else:
                ui.console.clear()
                ui.update_display()
                ui.console.print(ui.layout)
                
                user_input = ui.console.input("[cyan]Press ENTER for next question (or 'q' to quit)...[/cyan]")
                
                if user_input.lower() in ['q', 'quit']:
                    break
                
                # Clear feedback for next question
                ui.clear_feedback()
                app.invoke(None, config=config)
        
        else:
            app.invoke(None, config=config)
