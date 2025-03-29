import warnings
warnings.filterwarnings('ignore')

import os
import requests
import fitz  # PyMuPDF for PDF processing
import docx  # python-docx for DOCX processing
import gradio as gr
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Retrieve the Gemini API key from environment variables
gemini_key = os.getenv("geminiapikey")  # Ensure you have set the Gemini API key in your Hugging Face Space

# Other API keys (assuming they are also set in the Hugging Face Space environment)
os.environ["SERPER_API_KEY"] = os.getenv("serper_key")
os.environ["GRADIO_SERVER_PORT"] = "7862"

# Install necessary packages (ensure pymupdf and pydantic are installed)
os.system('pip install pydantic --upgrade')
os.system("pip install pymupdf")

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file using python-docx."""
    doc = docx.Document(file_path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return "\n".join(fullText)

# Function to determine file type and extract text from the resume
def extract_text_from_resume(file_path):
    """Determines file type and extracts text."""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return "Unsupported file format."

# Function to call the Gemini API for processing
def get_gemini_response(input_data):
    """Calls the Gemini API with the provided input."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=" + gemini_key # Replace with the actual Gemini API endpoint
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{
            "parts": [{
                "text": input_data
            }]
        }]
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text'] # Return the Gemini API response
    else:
        return f"Error fetching feedback from Gemini API. Status code: {response.status_code}, Response: {response.text}"

# Resume feedback agent
resume_feedback = Agent(
    role="Professional Resume Advisor",
    goal="Give feedback on the resume to make it stand out in the job market.",
    verbose=True,
    backstory="With a strategic mind and an eye for detail, you excel at providing feedback on resumes to highlight the most relevant skills and experiences."
)

# Task for Resume Strategist Agent: Align Resume with Job Requirements
resume_feedback_task = Task(
    description="""Give feedback on the resume to make it stand out for recruiters. Review every section, including the summary, work experience, skills, and education. Suggest to add relevant sections if they are missing. Also give an overall score to the resume out of 10.  This is the resume: {resume}""",
    expected_output="The overall score of the resume followed by the feedback in bullet points.",
    agent=resume_feedback
)

# Resume advisor agent
resume_advisor = Agent(
    role="Professional Resume Writer",
    goal="Based on the feedback received from Resume Advisor, make changes to the resume to make it stand out in the job market.",
    verbose=True,
    backstory="With a strategic mind and an eye for detail, you excel at refining resumes based on the feedback to highlight the most relevant skills and experiences."
)

# Task for Resume Advisor Agent: Align Resume with Job Requirements
resume_advisor_task = Task(
    description="""Rewrite the resume based on the feedback to make it stand out for recruiters. You can adjust and enhance the resume but don't make up facts. Review and update every section, including the summary, work experience, skills, and education to better reflect the candidate's abilities. This is the resume: {resume}""",
    expected_output="Resume in markdown format that effectively highlights the candidate's qualifications and experiences",
    context=[resume_feedback_task],
    agent=resume_advisor
)

# Job researcher agent
search_tool = SerperDevTool()

job_researcher = Agent(
    role="Senior Recruitment Consultant",
    goal="Find the 5 most relevant, recently posted jobs based on the improved resume received from resume advisor and the location preference",
    tools=[search_tool],
    verbose=True,
    backstory="As a senior recruitment consultant, your prowess in finding the most relevant jobs based on the resume and location preference is unmatched."
)

# Task for Job Researcher Agent
research_task = Task(
    description="""Find the 5 most relevant recent job postings based on the resume received from resume advisor and location preference. This is the preferred location: {location}. Use the tools to gather relevant content and shortlist the 5 most relevant, recent job openings""",
    expected_output="A bullet point list of the 5 job openings, with the appropriate links and detailed description about each job, in markdown format",
    agent=job_researcher
)

# Creating the Crew
crew = Crew(
    agents=[resume_feedback, resume_advisor, job_researcher],
    tasks=[resume_feedback_task, resume_advisor_task, research_task],
    verbose=True
)

# Function to run the entire agent pipeline
def resume_agent(file_path, location):
    resume_text = extract_text_from_resume(file_path)

    # Call the Gemini API to process the resume or provide feedback
    feedback = get_gemini_response(resume_feedback_task.description.format(resume = resume_text))
    improved_resume = get_gemini_response(resume_advisor_task.description.format(resume = resume_text))
    job_roles = get_gemini_response(research_task.description.format(location=location))

    return feedback, improved_resume, job_roles

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Resume Feedback and Job Matching Tool")
    gr.Markdown("*Expected Runtime: 1 Min*")
    
    with gr.Column():
        with gr.Row():
            resume_upload = gr.File(label="Upload Your Resume (PDF or DOCX)", height=120)
            location_input = gr.Textbox(label="Preferred Location", placeholder="e.g., Lahore")
            submit_button = gr.Button("Submit")
        
        with gr.Column():
            feedback_output = gr.Markdown(label="Resume Feedback")
            improved_resume_output = gr.Markdown(label="Improved Resume")
            job_roles_output = gr.Markdown(label="Relevant Job Roles")

    # Define the click event for the submit button
    def format_outputs(feedback, improved_resume, job_roles):
        feedback_with_heading = f"## Resume Feedback:\n\n{feedback}"
        improved_resume_with_heading = f"## Improved Resume:\n\n{improved_resume}"
        job_roles_with_heading = f"## Relevant Job Roles:\n\n{job_roles}"
        return feedback_with_heading, improved_resume_with_heading, job_roles_with_heading

    submit_button.click(
        lambda: gr.update(value="Processing..."),
        inputs=[],
        outputs=submit_button
    ).then(
        resume_agent,
        inputs=[resume_upload, location_input],
        outputs=[feedback_output, improved_resume_output, job_roles_output]
    ).then(
        format_outputs,
        inputs=[feedback_output, improved_resume_output, job_roles_output],
        outputs=[feedback_output, improved_resume_output, job_roles_output]
    ).then(
        lambda: gr.update(value="Submit"),
        inputs=[],
        outputs=submit_button
    )

# Launch the Gradio app
demo.queue()
demo.launch()