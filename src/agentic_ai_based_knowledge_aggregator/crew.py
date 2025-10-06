from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from crewai_tools import SerperDevTool

import os
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model=os.getenv("OPENROUTER_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL")
)

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
            Vector db tool: if user has provided any pdf, txt, json or anything make a vector db of that for reference

        Returns:
            Agent: crew agent
        """
        return Agent(
            config=self.agents_config['search_agent'],
            verbose=True,
            llm=llm,
            tools=[SerperDevTool()]
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

    @crew
    def crew(self) -> Crew:
        """Creates the AgenticAiBasedKnowledgeAggregator crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
