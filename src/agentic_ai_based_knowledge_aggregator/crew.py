from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from crewai_tools import SerperDevTool

import os
from pathlib import Path
from dotenv import load_dotenv
from tools.document_knowledge_tool import DocumentKnowledgeTool

load_dotenv()

llm = LLM(
    model=os.getenv("OPENROUTER_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL")
)

# , "How is python different from other languages", "What are basic constituents of python", "Simple programs with explanation in python", "Must know about python"
# llm = LLM(
#     model="ollama/tinydolphin:latest",
#     base_url="http://localhost:11434"
# )


@CrewBase
class AgenticAiBasedKnowledgeAggregator():
    """AgenticAiBasedKnowledgeAggregator crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def initializer(self) -> Agent:
        """Decode the requirements of the user,
           generate a list of specified no. of topics for web search (default=5)

        Returns:
            Agent: crew agent
        """
        return Agent(
            config=self.agents_config['initializer'],
            verbose=True,
            llm=llm
        )

    @agent
    def search_agent(self) -> Agent:
        """Search web about the topic list from initializer,
           and store summarized (by llm) data in a txt file
           (citations if possible)
        
        Tools:
            Serper Tool: for web search
            file tool: to read data from user provided file
        Returns:
            Agent: crew agent
        """
        serper_tool = SerperDevTool()
        file_tool = DocumentKnowledgeTool()

        return Agent(
            config=self.agents_config['search_agent'],
            verbose=True,
            llm=llm,
            tools=[serper_tool, file_tool]
            # tools=[file_tool]   # for testing the file tool
        )
    
    @agent
    def reviewer_agent(self) -> Agent:
        """Review the data from search_agent, remove duplicates and create well curated chunks

        Returns:
            Agent: crew agent
        """

        return Agent(
            config=self.agents_config['reviewer_agent'],
            verbose=True,
            llm=llm,
        )

    @task
    def initializer_task(self) -> Task:
        return Task(
            config=self.tasks_config['initializer_task']
        )
    
    @task
    def search_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_agent_task']
        )

    @task
    def reviewer_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['reviewer_agent_task']
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AgenticAiBasedKnowledgeAggregator crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True
        )
