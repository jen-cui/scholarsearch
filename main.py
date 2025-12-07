from src.agent_system import Agent
from transformers import pipeline

llm = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def scholarsearch(prompt):
    return Agent(llm).run(prompt)

scholarsearch("What are the effects of e-cigarettes on sleep quality?")