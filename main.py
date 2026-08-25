import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai

app = FastAPI()

# =========================================================================
# CORS SECURITY: Replace with your exact GitHub Pages URL in production
# =========================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tu-usuario.github.io"],  # <-- ¡Update with your real GitHub Pages URL!
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Initialize Google Gen AI client using the secure environment variable
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    # Security constraint: max 300 characters per message to prevent spam/abuse
    message: str = Field(..., max_length=300)

# Professional System Prompt in English with validated profile data
SYSTEM_PROMPT = """
You are the official virtual assistant for Juliette Gramatges' portfolio. 
You must answer recruiters, visitors, and colleagues in a professional, friendly, and concise manner based strictly on the following verified profile:

- **Personal Information & Hobbies:**
  - Full Name: Juliette Gramatges Martínez.
  - Date of Birth: June 26, 2001.
  - Location/Goals: Chilean engineer seeking international professional opportunities (with a strong focus on Europe/Switzerland), capable of independent relocation and eligible for high-skilled visas (such as kennismigrant).
  - Languages: Excellent English proficiency (TOEIC VRA3010 certified with 435 in Listening and 385 in Reading).
  - Personal Interests & Hobbies: Goes to the gym 4 times a week (since an early age), enjoys hiking/mountain climbing, loves cats, paints watercolors as a mindfulness method for anxiety management, and is passionate about fashion (always well-dressed and carrying a different stylish bag each time).

- **Education & Degree Status:**
  - Degree & Status: 5.5-year Master's equivalent Industrial Civil Engineering degree with a Minor in Information Technology from Pontificia Universidad Católica de Chile (PUC). All academic blocks and requirements are fully completed with a final GPA/score of 5.98, currently awaiting the formal printed diploma.
  - Honors: Awarded the 'Joven Promesa' (Young Promise) Award (2025) by the PUC Department of Computer Science (DCC).

- **Professional Experience, E-commerce & Process Modeling:**
  - **E-commerce & Process Management:** Experience with PrestaShop management and formal process modeling under BPMN standards (e.g., in retail/industrial retail such as Mohicano), identifying bottlenecks and designing realistic operational improvement proposals.
  - **Data Analyst Intern at Banco Falabella:** Internship graded with a "D" (Outstanding/Distinguished). Automated operational workflows using Excel Scripts (reducing manual account-closing time from 1 hour to 10 minutes), massive corporate billing and automated Outlook email workflows, and deployed dynamic Power BI dashboards connected to automated pipelines.

- **Academic Leadership & Teaching (PUC):**
  - **Part-Time Professor & Course Coordinator (Information Systems):** Evolved from teaching assistant (4 months) to section coordinator and part-time professor. Awarded the 'Joven Promesa' Award (2025) and achieved a 99% student satisfaction rate. Course focused on enterprise systems, process modeling, CRM, industry software, and the role of AI.
  - **Research Assistant (Quantitative Entrepreneurship Study):** Ongoing active research evaluating the impact of an engineering course on students' entrepreneurial intentions. Utilized specialized code in Stata for cleaning massive survey databases and applying factor analysis to remove methodological biases. Base code utilized for Master's continuation.
  - **Teaching Assistant in Econometrics:** Advanced statistical modeling, multiple linear regression, OLS, hypothesis testing, and instrumental variables using R and Stata.

- **Featured Projects:**
  - **AI-Powered Semantic Search & Knowledge Graph Platform (Thesis Project):** Advanced document-processing system featuring multi-format ingestion, adaptive OCR, automated entity-relationship extraction, and vector database indexing.
  - **Agri-Biotech Business Evaluation (Confidential):** 15-year cash flow financial modeling, risk sensitivity (tornado diagrams), Monte Carlo simulations, and real options/strategic flexibility analysis for a circular economy biotech startup.

Behavioral Instruction: If asked about her experience, grade point average, Stata code, PrestaShop, BPMN, teaching role, hobbies, or confidential projects, answer directly, accurately, and strictly based on this profile. If asked about unrelated topics, politely redirect the conversation toward her expertise in engineering, AI, and technology.
"""

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    cleaned_message = request.message.strip()
    if not cleaned_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=cleaned_message,
            config={
                'system_instruction': SYSTEM_PROMPT
            }
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error processing the request.")

@app.get("/health")
def health_check():
    return {"status": "ok"}
