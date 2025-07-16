import os
from crewai import Agent, Task, Crew, LLM
from langchain_groq import ChatGroq
import gradio as gr

# Set API keys
groq_api_key = ""
os.environ["GROQ_API_KEY"] = groq_api_key
hf_api_key = ""
os.environ["HUGGINGFACEHUB_API_KEY"] = hf_api_key
serper_api_key = ""
os.environ["SERPER_API_KEY"] = serper_api_key

# Initialize LLM
groq_llm = LLM(
    model="groq/llama3-8b-8192",
    temperature=0.3,
    max_tokens=4096,
    api_key=groq_api_key,
)

# Create Agents
healthcare_assistant = Agent(
    role="Healthcare Assistant",
    goal="Research and provide accurate healthcare information",
    backstory="""You are an experienced healthcare information specialist with extensive
    knowledge of medical conditions, treatments, and preventive care. You provide accurate,
    well-researched information from reliable medical sources.""",
    llm=groq_llm,
    verbose=True
)

reviewer = Agent(
    role="Medical Reviewer",
    goal="Review and verify medical information for accuracy",
    backstory="""You are a senior medical professional who reviews healthcare information
    for accuracy, completeness, and proper medical disclaimers. You ensure all information
    follows current medical guidelines.""",
    llm=groq_llm,
    verbose=True
)

def process_health_query(query):
    # Create tasks
    research_task = Task(
        description=f"""Research this health query thoroughly: {query}
        1. Focus on reliable medical sources
        2. Include symptoms, causes, and treatments if applicable
        3. Add relevant medical context
        4. Note any important warnings or considerations""",
        agent=healthcare_assistant,
        expected_output="Detailed medical information from reliable sources"
    )

    review_task = Task(
        description=f"""Review this medical information for accuracy and completeness:
        1. Verify all medical facts
        2. Ensure appropriate medical disclaimers
        3. Check for clarity and completeness
        4. Add any missing critical information""",
        agent=reviewer,
        expected_output="Verified and complete medical information with appropriate disclaimers"
    )

    # Create crew
    crew = Crew(
        agents=[healthcare_assistant, reviewer],
        tasks=[research_task, review_task],
        verbose=True
    )

    # Process the query
    result = crew.kickoff()
    return result

def gradio_interface(query):
    try:
        if not query.strip():
            return "Please enter a valid question."
        return process_health_query(query)
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create and launch Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter your healthcare question here...",
        label="Question"
    ),
    outputs=gr.Textbox(
        label="Medical Response",
        lines=10
    ),
    title="Healthcare Assistant",
    description="Ask any healthcare-related questions and receive accurate, reviewed responses.",
    examples=[
        ["What are the common symptoms of diabetes?"],
        ["How can I prevent heart disease?"],
        ["What are the side effects of high blood pressure medication?"]
    ]
)

# For Colab, use this line
interface.launch(share=True, debug=True)
