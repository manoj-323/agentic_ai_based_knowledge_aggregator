from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."


class web_search_tool_input(BaseModel):
    """Input schema for web search of queries"""
    argument: str = Field(..., description="The web search query.")

class web_search_tool(BaseTool):
    name: str = "custom_web_search_tool"
    description: str = (
        "This tool is used for searching the web for any query and it return the json response for that query."
    )
    args_schema: Type[BaseModel] = web_search_tool_input

    def _run(self, argument: str) -> str:
        pass