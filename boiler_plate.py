from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from .env
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)
app = FastAPI(title="Email Rewriter Assistant")

# Input and output schemas
class EmailInput(BaseModel):
    draft: str

class EmailOutput(BaseModel):
    polished: str

# Rewriting function
def rewrite_email(draft: str) -> str:
    prompt = (
        f"Rewrite the following email to be more professional, polished and clear "
        f"while keeping the original message and intent:\n\n{draft}\n\nRewritten Email:"
    )

    try:
        # No streaming to simplify
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stop=None,
        )
        return completion.choices[0].message.content.strip()

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail="Failed to rewrite email")

# API endpoint
@app.post("/rewrite", response_model=EmailOutput)
def rewrite(email: EmailInput):
    rewritten = rewrite_email(email.draft)
    return {"polished": rewritten}  
