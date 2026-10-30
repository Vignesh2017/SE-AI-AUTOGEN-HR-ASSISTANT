
import asyncio
import os

from dotenv import load_dotenv 
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

async def main() -> None:
    # 1. Read the OpenAI API key from an environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not configured."
        )

    # 2. Create the OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=api_key,
    )

    try:
        # 3. Create the HR Assistant agent
        agent = AssistantAgent(
            name="hr_assistant",
            model_client=model_client,
            system_message="You are an HR assistant.",
            model_client_stream=True,
        )

        # 4. Ask the agent to generate a five-line job description
        task = (
            "Write a job description for a Junior Python Developer. "
            "The description must contain exactly 5 lines, covering "
            "the role, responsibilities, technical skills, qualifications, "
            "and preferred experience. Do not add an introduction."
        )

        # 5. Run the agent with streaming and display its response
        print("\n=== HR Assistant: Job Description ===\n")

        await Console(
            agent.run_stream(task=task),
            output_stats=True,
        )

    finally:
        # 6. Close the client to release resources
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
