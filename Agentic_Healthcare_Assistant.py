"""
Agentic Healthcare Assistant for Medical Task Automation
=========================================================
Purdue Generative AI Specialist – Capstone Project

A multi-agent GenAI system that handles:
  • Appointment scheduling with doctor availability
  • Patient medical record management
  • Medical history retrieval via RAG (FAISS)
  • Disease information search and summarization

Tech stack: LangGraph · GPT-4o-mini · FAISS (RAG) · SQLite · Streamlit
"""

import os
import re
import sys
import random
import sqlite3
import textwrap
import pandas as pd
import streamlit as st
from typing import TypedDict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from pypdf import PdfReader

# ── Environment ──
load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "healthcare.db")
PDF_DIR = os.path.join(BASE_DIR, "Reference Materials",
                       "Agentic Healthcare Assistant for Medical Task Automation")


# ============================================================================
#  SECTION 1 — DATABASE LAYER (SQLite)
# ============================================================================

def get_connection():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables and seed with sample data."""
    conn = get_connection()
    c = conn.cursor()

    # ── Patients ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            phone        TEXT,
            email        TEXT,
            name         TEXT NOT NULL,
            age          INTEGER,
            gender       TEXT,
            address      TEXT,
            summary      TEXT
        )
    """)

    # ── Doctors ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            doctor_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            specialty    TEXT NOT NULL,
            available_days TEXT NOT NULL,
            available_hours TEXT NOT NULL
        )
    """)

    # ── Appointments ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name   TEXT NOT NULL,
            doctor_name    TEXT NOT NULL,
            specialty      TEXT NOT NULL,
            date           TEXT NOT NULL,
            time_slot      TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'Confirmed',
            created_at     TEXT NOT NULL
        )
    """)

    # Seed patients if empty
    c.execute("SELECT COUNT(*) FROM patients")
    if c.fetchone()[0] == 0:
        patients = [
            ("7982179305", "rahul16negi@gmail.com", "Rahul Negi", 31, "Male",
             "Chattarpur", "Rahul is a fit and healthy person. He is doing well in his life"),
            ("+91-98220-45322", None, "Ramesh Kulkarni", 65, "Male",
             "52 Residency Road, Chennai",
             "Patient presents for routine checkup with a history of hypertension. "
             "Vitals are stable. Continue current medication and recommend lifestyle "
             "modifications. Routine labs ordered and return in 6 months."),
            ("+91-98180-11245", None, "Anjali Mehra", 33, "Female",
             "202 Lakeview Apartments, Pune",
             "Patient presents with 5-day history of dry cough and mild fever, diagnosed "
             "with Upper Respiratory Infection (J06.9), and advised symptomatic management "
             "with antihistamines and fluids, rest, and follow-up in 5 days."),
            ("+91-98450-11223", None, "David Thompson", 51, "Male",
             "17 MG Road, Indiranagar, Bangalore",
             "Patient presents for follow-up of Type 2 Diabetes. Reports increased thirst "
             "and urination. Diagnosis: Type 2 Diabetes Mellitus (E11.9). Plan: Increase "
             "metformin dosage, order HbA1c and Lipid Profile, schedule nutritionist "
             "follow-up, return in 3 months."),
        ]
        c.executemany(
            "INSERT INTO patients (phone, email, name, age, gender, address, summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", patients)

    # Seed doctors if empty
    c.execute("SELECT COUNT(*) FROM doctors")
    if c.fetchone()[0] == 0:
        doctors = [
            ("Dr. Priya Sharma", "General Physician", "Mon,Tue,Wed,Thu,Fri",
             "09:00-12:00,14:00-17:00"),
            ("Dr. Amit Patel", "Cardiologist", "Mon,Wed,Fri",
             "10:00-13:00,15:00-18:00"),
            ("Dr. Kavitha Rao", "Nephrologist", "Tue,Thu,Sat",
             "09:00-12:00,14:00-16:00"),
            ("Dr. Suresh Menon", "Endocrinologist", "Mon,Wed,Thu,Fri",
             "08:00-11:00,13:00-16:00"),
        ]
        c.executemany(
            "INSERT INTO doctors (name, specialty, available_days, available_hours) "
            "VALUES (?, ?, ?, ?)", doctors)

    # Seed appointments if empty
    c.execute("SELECT COUNT(*) FROM appointments")
    if c.fetchone()[0] == 0:
        sample_appts = [
            ("Ramesh Kulkarni", "Dr. Priya Sharma", "General Physician",
             (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
             "10:00", "Confirmed", datetime.now().isoformat()),
            ("David Thompson", "Dr. Suresh Menon", "Endocrinologist",
             (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
             "09:00", "Confirmed", datetime.now().isoformat()),
        ]
        c.executemany(
            "INSERT INTO appointments (patient_name, doctor_name, specialty, date, "
            "time_slot, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", sample_appts)

    conn.commit()
    conn.close()


def get_all_patients() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM patients ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient_by_name(name: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM patients WHERE LOWER(name) LIKE ?",
        (f"%{name.lower()}%",)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def add_or_update_patient(name: str, age: int = None, gender: str = None,
                          phone: str = None, address: str = None,
                          summary: str = None) -> str:
    existing = get_patient_by_name(name)
    conn = get_connection()
    if existing:
        updates = []
        params = []
        if age: updates.append("age=?"); params.append(age)
        if gender: updates.append("gender=?"); params.append(gender)
        if phone: updates.append("phone=?"); params.append(phone)
        if address: updates.append("address=?"); params.append(address)
        if summary: updates.append("summary=?"); params.append(summary)
        if updates:
            params.append(existing["patient_id"])
            conn.execute(f"UPDATE patients SET {','.join(updates)} WHERE patient_id=?", params)
            conn.commit()
        conn.close()
        return f"Updated record for {name}."
    else:
        conn.execute(
            "INSERT INTO patients (name, age, gender, phone, address, summary) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, age, gender, phone, address, summary))
        conn.commit()
        conn.close()
        return f"Created new record for {name}."


def get_all_doctors() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM doctors ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_doctor_by_specialty(specialty: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM doctors WHERE LOWER(specialty) LIKE ?",
        (f"%{specialty.lower()}%",)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def book_appointment(patient_name: str, doctor_name: str, specialty: str,
                     date: str, time_slot: str) -> str:
    conn = get_connection()
    # Check for conflicting appointment
    existing = conn.execute(
        "SELECT 1 FROM appointments WHERE doctor_name=? AND date=? AND time_slot=? AND status='Confirmed'",
        (doctor_name, date, time_slot)
    ).fetchone()
    if existing:
        conn.close()
        return f"Sorry, {doctor_name} is already booked at {time_slot} on {date}."

    conn.execute(
        "INSERT INTO appointments (patient_name, doctor_name, specialty, date, time_slot, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'Confirmed', ?)",
        (patient_name, doctor_name, specialty, date, time_slot, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return (f"Appointment booked for {patient_name} with {doctor_name} ({specialty}) "
            f"on {date} at {time_slot}.")


def get_all_appointments() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM appointments ORDER BY date DESC, time_slot").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize database
init_db()


# ============================================================================
#  SECTION 2 — RAG PIPELINE (FAISS over PDF medical reports)
# ============================================================================

@st.cache_resource
def build_vector_store():
    """Load PDF medical reports, chunk, embed, and build FAISS index."""
    documents = []

    # Load PDFs from the reference materials directory
    pdf_files = [
        "sample_report_anjali.pdf",
        "sample_report_david.pdf",
        "sample_report_ramesh.pdf",
    ]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        if os.path.exists(pdf_path):
            reader = PdfReader(pdf_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            # Extract patient name from filename
            patient = pdf_file.replace("sample_report_", "").replace(".pdf", "").title()
            documents.append(Document(
                page_content=text,
                metadata={"source": pdf_file, "patient": patient}
            ))

    # Also add patient summaries from the database as documents
    for patient in get_all_patients():
        if patient.get("summary"):
            documents.append(Document(
                page_content=(
                    f"Patient: {patient['name']}\n"
                    f"Age: {patient.get('age', 'N/A')}, Gender: {patient.get('gender', 'N/A')}\n"
                    f"Summary: {patient['summary']}"
                ),
                metadata={"source": "patient_db", "patient": patient["name"]}
            ))

    if not documents:
        return None

    # Chunk the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Build FAISS index with OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def retrieve_patient_context(query: str, k: int = 4) -> str:
    """Retrieve relevant chunks from FAISS for a given query."""
    vs = build_vector_store()
    if vs is None:
        return "No medical records available in the vector store."
    docs = vs.similarity_search(query, k=k)
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


# ============================================================================
#  SECTION 3 — LLM SETUP
# ============================================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


# ============================================================================
#  SECTION 4 — AGENT STATE SCHEMA
# ============================================================================

class AgentState(TypedDict):
    user_message: str
    intent: str             # "appointment" | "record_update" | "history" | "disease_search"
    patient_name: str
    response: str
    logs: list


# ============================================================================
#  SECTION 5 — AGENT 1: PLANNER (Intent Classification + Goal Decomposition)
# ============================================================================

_planner_prompt = PromptTemplate(
    input_variables=["user_message"],
    template=textwrap.dedent("""\
        You are a medical assistant planner. Classify the patient's request
        into EXACTLY one intent category:

        - appointment : The user wants to book, schedule, or check a doctor appointment.
        - record_update : The user wants to add, update, or modify patient medical records.
        - history : The user wants to retrieve, view, or summarize past medical history.
        - disease_search : The user wants information about a disease, condition, or treatment.

        Patient request: {user_message}

        Respond with ONLY the intent category name. Nothing else."""),
)
_planner_chain = _planner_prompt | llm | StrOutputParser()


def planner_node(state: AgentState) -> dict:
    """Classify user intent and decompose into sub-goals."""
    raw = _planner_chain.invoke({"user_message": state["user_message"]})
    intent = raw.strip().lower().replace(" ", "_")

    valid = {"appointment", "record_update", "history", "disease_search"}
    if intent not in valid:
        if "appoint" in intent or "book" in intent or "schedule" in intent:
            intent = "appointment"
        elif "record" in intent or "update" in intent or "add" in intent:
            intent = "record_update"
        elif "history" in intent or "summar" in intent or "retrieve" in intent:
            intent = "history"
        else:
            intent = "disease_search"

    log_entry = {
        "agent": "Planner",
        "timestamp": datetime.now().isoformat(),
        "input": state["user_message"][:120],
        "output": intent,
        "status": "success",
    }
    return {
        "intent": intent,
        "logs": state.get("logs", []) + [log_entry],
    }


# ============================================================================
#  SECTION 6 — AGENT 2: APPOINTMENT BOOKING
# ============================================================================

_appointment_prompt = PromptTemplate(
    input_variables=["user_message", "doctors_info", "existing_appointments"],
    template=textwrap.dedent("""\
        You are a medical appointment scheduling assistant.

        Available doctors:
        {doctors_info}

        Existing appointments:
        {existing_appointments}

        Patient request: {user_message}

        Based on the request, determine:
        1. The patient name (if mentioned)
        2. The required specialty or doctor name
        3. A suitable date (use upcoming weekdays from today)
        4. A suitable time slot from the doctor's available hours

        Respond in this EXACT format (one item per line):
        PATIENT: <name>
        DOCTOR: <doctor name>
        SPECIALTY: <specialty>
        DATE: <YYYY-MM-DD>
        TIME: <HH:MM>
        SUMMARY: <brief description of what was booked>

        If you cannot determine the patient name, use "Unknown Patient".
        Pick the next available date that matches the doctor's available days."""),
)
_appointment_chain = _appointment_prompt | llm | StrOutputParser()


def appointment_node(state: AgentState) -> dict:
    """Book an appointment based on the user request."""
    doctors = get_all_doctors()
    doctors_info = "\n".join(
        f"- {d['name']} ({d['specialty']}): Available {d['available_days']}, "
        f"Hours: {d['available_hours']}"
        for d in doctors
    )

    appts = get_all_appointments()
    appts_info = "\n".join(
        f"- {a['patient_name']} with {a['doctor_name']} on {a['date']} at {a['time_slot']}"
        for a in appts[:10]
    ) if appts else "No existing appointments."

    result = _appointment_chain.invoke({
        "user_message": state["user_message"],
        "doctors_info": doctors_info,
        "existing_appointments": appts_info,
    })

    # Parse the structured response
    patient = doctor = specialty = date = time = summary = ""
    for line in result.strip().split("\n"):
        line = line.strip()
        if line.startswith("PATIENT:"):
            patient = line.split(":", 1)[1].strip()
        elif line.startswith("DOCTOR:"):
            doctor = line.split(":", 1)[1].strip()
        elif line.startswith("SPECIALTY:"):
            specialty = line.split(":", 1)[1].strip()
        elif line.startswith("DATE:"):
            date = line.split(":", 1)[1].strip()
        elif line.startswith("TIME:"):
            time = line.split(":", 1)[1].strip()
        elif line.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    # Book the appointment in the database
    if doctor and date and time:
        booking_result = book_appointment(patient or "Unknown Patient", doctor,
                                          specialty, date, time)
        response = f"📅 {booking_result}\n\n**Details:**\n- Patient: {patient}\n- Doctor: {doctor} ({specialty})\n- Date: {date}\n- Time: {time}"
    else:
        response = (
            f"I'd like to help you book an appointment. Here are our available doctors:\n\n"
            + "\n".join(f"- **{d['name']}** ({d['specialty']}) — {d['available_days']}"
                        for d in doctors)
            + "\n\nPlease specify the doctor or specialty, and I'll find the next available slot."
        )

    log_entry = {
        "agent": "Appointment Agent",
        "timestamp": datetime.now().isoformat(),
        "action": f"booking: {patient} → {doctor} on {date}",
        "input": state["user_message"][:120],
        "output": response[:150],
        "status": "success" if doctor and date else "partial",
    }
    return {
        "patient_name": patient,
        "response": response,
        "logs": state.get("logs", []) + [log_entry],
    }


# ============================================================================
#  SECTION 7 — AGENT 3: MEDICAL RECORDS MANAGEMENT
# ============================================================================

_records_prompt = PromptTemplate(
    input_variables=["user_message", "existing_patients"],
    template=textwrap.dedent("""\
        You are a medical records management assistant.

        Existing patients in the database:
        {existing_patients}

        User request: {user_message}

        Extract the following from the request:
        1. Patient name
        2. Any updates to: age, gender, phone, address, or medical summary

        Respond in this EXACT format:
        NAME: <patient name>
        AGE: <age or NONE>
        GENDER: <gender or NONE>
        PHONE: <phone or NONE>
        ADDRESS: <address or NONE>
        SUMMARY: <medical summary update or NONE>
        ACTION: <brief description of what was updated>"""),
)
_records_chain = _records_prompt | llm | StrOutputParser()


def records_node(state: AgentState) -> dict:
    """Add or update patient records."""
    patients = get_all_patients()
    patients_info = "\n".join(
        f"- {p['name']} (Age: {p.get('age','N/A')}, {p.get('gender','N/A')})"
        for p in patients
    )

    result = _records_chain.invoke({
        "user_message": state["user_message"],
        "existing_patients": patients_info,
    })

    # Parse the response
    name = age = gender = phone = address = summary = action = None
    for line in result.strip().split("\n"):
        line = line.strip()
        if line.startswith("NAME:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("AGE:"):
            val = line.split(":", 1)[1].strip()
            age = int(val) if val.isdigit() else None
        elif line.startswith("GENDER:"):
            val = line.split(":", 1)[1].strip()
            gender = val if val.upper() != "NONE" else None
        elif line.startswith("PHONE:"):
            val = line.split(":", 1)[1].strip()
            phone = val if val.upper() != "NONE" else None
        elif line.startswith("ADDRESS:"):
            val = line.split(":", 1)[1].strip()
            address = val if val.upper() != "NONE" else None
        elif line.startswith("SUMMARY:"):
            val = line.split(":", 1)[1].strip()
            summary = val if val.upper() != "NONE" else None
        elif line.startswith("ACTION:"):
            action = line.split(":", 1)[1].strip()

    if name:
        db_result = add_or_update_patient(name, age, gender, phone, address, summary)
        response = f"📋 {db_result}\n\n**Action:** {action or 'Record updated'}"
    else:
        response = "I couldn't identify the patient name. Please specify whose record to update."

    log_entry = {
        "agent": "Records Agent",
        "timestamp": datetime.now().isoformat(),
        "action": action or "record_update",
        "input": state["user_message"][:120],
        "output": response[:150],
        "status": "success" if name else "failure",
    }
    return {
        "patient_name": name or "",
        "response": response,
        "logs": state.get("logs", []) + [log_entry],
    }


# ============================================================================
#  SECTION 8 — AGENT 4: HISTORY RETRIEVAL (RAG)
# ============================================================================

_history_prompt = PromptTemplate(
    input_variables=["user_message", "retrieved_context"],
    template=textwrap.dedent("""\
        You are a medical history summarization assistant.
        Use the retrieved patient records to answer the query.

        Retrieved medical records:
        {retrieved_context}

        Patient query: {user_message}

        Provide a clear, structured summary of the relevant medical history.
        Include: diagnoses, treatments, vitals, and any recommendations.
        If the records don't contain relevant information, say so clearly.
        Keep the response concise and professional."""),
)
_history_chain = _history_prompt | llm | StrOutputParser()


def history_node(state: AgentState) -> dict:
    """Retrieve and summarize patient medical history via RAG."""
    user_message = state["user_message"]

    # Retrieve context from FAISS
    context = retrieve_patient_context(user_message)

    # Generate summary using LLM + retrieved context
    response = _history_chain.invoke({
        "user_message": user_message,
        "retrieved_context": context,
    })

    log_entry = {
        "agent": "History Agent (RAG)",
        "timestamp": datetime.now().isoformat(),
        "action": "rag_retrieval + summarization",
        "input": user_message[:120],
        "output": response[:150],
        "rag_chunks": len(context.split("---")),
        "status": "success",
    }
    return {
        "response": response,
        "logs": state.get("logs", []) + [log_entry],
    }


# ============================================================================
#  SECTION 9 — AGENT 5: DISEASE SEARCH
# ============================================================================

_disease_prompt = PromptTemplate(
    input_variables=["user_message"],
    template=textwrap.dedent("""\
        You are a medical information assistant providing evidence-based health information.
        You draw from medical knowledge including sources like Medline and WHO guidelines.

        Patient query: {user_message}

        Provide a comprehensive but concise response covering:
        1. **Overview** of the condition/disease
        2. **Common symptoms**
        3. **Standard treatments** and management approaches
        4. **When to seek medical attention**

        Important: Include a disclaimer that this is for informational purposes only
        and the patient should consult their doctor for personalized medical advice.
        Keep the response professional and easy to understand."""),
)
_disease_chain = _disease_prompt | llm | StrOutputParser()


def disease_search_node(state: AgentState) -> dict:
    """Provide disease information using LLM medical knowledge."""
    response = _disease_chain.invoke({"user_message": state["user_message"]})

    log_entry = {
        "agent": "Disease Search Agent",
        "timestamp": datetime.now().isoformat(),
        "action": "disease_info_search",
        "input": state["user_message"][:120],
        "output": response[:150],
        "status": "success",
    }
    return {
        "response": response,
        "logs": state.get("logs", []) + [log_entry],
    }


# ============================================================================
#  SECTION 10 — ROUTING FUNCTION
# ============================================================================

def route_by_intent(state: AgentState) -> str:
    """Conditional edge: route to the correct agent based on intent."""
    intent = state.get("intent", "disease_search")
    mapping = {
        "appointment": "appointment_agent",
        "record_update": "records_agent",
        "history": "history_agent",
        "disease_search": "disease_search_agent",
    }
    return mapping.get(intent, "disease_search_agent")


# ============================================================================
#  SECTION 11 — LANGGRAPH WORKFLOW
# ============================================================================

def build_graph():
    """Construct and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("appointment_agent", appointment_node)
    graph.add_node("records_agent", records_node)
    graph.add_node("history_agent", history_node)
    graph.add_node("disease_search_agent", disease_search_node)

    # Entry point
    graph.set_entry_point("planner")

    # Conditional routing from planner
    graph.add_conditional_edges(
        "planner",
        route_by_intent,
        {
            "appointment_agent": "appointment_agent",
            "records_agent": "records_agent",
            "history_agent": "history_agent",
            "disease_search_agent": "disease_search_agent",
        },
    )

    # Terminal edges
    graph.add_edge("appointment_agent", END)
    graph.add_edge("records_agent", END)
    graph.add_edge("history_agent", END)
    graph.add_edge("disease_search_agent", END)

    return graph.compile()


app_graph = build_graph()


def run_agent(user_message: str) -> dict:
    """Execute the full multi-agent pipeline."""
    initial_state: AgentState = {
        "user_message": user_message,
        "intent": "",
        "patient_name": "",
        "response": "",
        "logs": [],
    }
    return app_graph.invoke(initial_state)


# ============================================================================
#  SECTION 12 — MODEL EVALUATION
# ============================================================================

_eval_llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

_grade_prompt = PromptTemplate(
    input_variables=["intent", "expected_intent", "response", "criteria"],
    template=textwrap.dedent("""\
        You are an evaluator grading an AI healthcare assistant.

        The agent classified a patient request as: {intent}
        The expected classification was: {expected_intent}

        The agent response was:
        {response}

        Evaluation criteria: {criteria}

        Grade the response as CORRECT or INCORRECT.
        - CORRECT if the intent matches AND the response meets the criteria.
        - INCORRECT otherwise.

        Respond with ONLY "CORRECT" or "INCORRECT"."""),
)
_grade_chain = _grade_prompt | _eval_llm | StrOutputParser()

TEST_CASES = [
    # ── Appointments ──
    {
        "message": "I need to book a nephrologist appointment for Ramesh Kulkarni.",
        "expected_intent": "appointment",
        "criteria": "Response should confirm booking with a nephrologist and mention Ramesh Kulkarni.",
    },
    {
        "message": "Schedule a follow-up with an endocrinologist for David Thompson next week.",
        "expected_intent": "appointment",
        "criteria": "Response should book an endocrinologist appointment for David Thompson.",
    },
    {
        "message": "Can I get an appointment with Dr. Priya Sharma for a general checkup?",
        "expected_intent": "appointment",
        "criteria": "Response should reference Dr. Priya Sharma and schedule an appointment.",
    },
    # ── Record Updates ──
    {
        "message": "Update Anjali Mehra's record: she is now 34 years old and has moved to Mumbai.",
        "expected_intent": "record_update",
        "criteria": "Response should confirm updating age to 34 and address to Mumbai.",
    },
    {
        "message": "Add a new patient: Sarah Khan, 28, Female, phone +91-98765-43210.",
        "expected_intent": "record_update",
        "criteria": "Response should confirm creating a new patient record for Sarah Khan.",
    },
    # ── History Retrieval ──
    {
        "message": "Can you summarize Anjali Mehra's medical history?",
        "expected_intent": "history",
        "criteria": "Response should mention upper respiratory infection, dry cough, fever from her records.",
    },
    {
        "message": "What is David Thompson's current diagnosis and treatment plan?",
        "expected_intent": "history",
        "criteria": "Response should mention Type 2 Diabetes, metformin, HbA1c from his records.",
    },
    {
        "message": "Retrieve Ramesh Kulkarni's latest health summary.",
        "expected_intent": "history",
        "criteria": "Response should mention hypertension and Telmisartan from his records.",
    },
    # ── Disease Search ──
    {
        "message": "What are the latest treatment methods for chronic kidney disease?",
        "expected_intent": "disease_search",
        "criteria": "Response should cover CKD overview, symptoms, treatments, and include a disclaimer.",
    },
    {
        "message": "Tell me about Type 2 Diabetes management and prevention.",
        "expected_intent": "disease_search",
        "criteria": "Response should cover diabetes management, lifestyle changes, medication, and disclaimer.",
    },
]


def run_evaluation() -> pd.DataFrame:
    """Run all test cases and grade each result."""
    results = []
    for i, tc in enumerate(TEST_CASES, 1):
        try:
            output = run_agent(tc["message"])
            actual_intent = output.get("intent", "unknown")
            response = output.get("response", "")
            grade = _grade_chain.invoke({
                "intent": actual_intent,
                "expected_intent": tc["expected_intent"],
                "response": response,
                "criteria": tc["criteria"],
            }).strip()
        except Exception as e:
            actual_intent = "error"
            response = str(e)
            grade = "INCORRECT"

        results.append({
            "Test #": i,
            "Expected": tc["expected_intent"],
            "Actual": actual_intent,
            "Response": response[:180] + "..." if len(response) > 180 else response,
            "Grade": grade,
        })
    return pd.DataFrame(results)


def compute_metrics(eval_df: pd.DataFrame) -> dict:
    total = len(eval_df)
    correct = (eval_df["Grade"] == "CORRECT").sum()
    routing = (eval_df["Expected"] == eval_df["Actual"]).sum()
    return {
        "total_tests": total,
        "correct": int(correct),
        "incorrect": int(total - correct),
        "accuracy": f"{correct / total * 100:.0f}%",
        "routing_accuracy": f"{routing / total * 100:.0f}%",
    }


# ============================================================================
#  SECTION 13 — STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="🏥 HealthAgent AI – Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a2744 50%, #0d1117 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1e38 0%, #1a2744 100%);
        border-right: 1px solid rgba(16, 185, 129, 0.15);
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(20, 40, 70, 0.9), rgba(15, 30, 55, 0.9));
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #10b981; }
    .kpi-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .user-bubble {
        background: linear-gradient(135deg, #1e3a5f, #1e40af);
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px; margin: 8px 0;
        color: #e2e8f0; max-width: 80%; margin-left: auto;
    }
    .agent-bubble {
        background: linear-gradient(135deg, #1a2744, #1e293b);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px; margin: 8px 0;
        color: #e2e8f0; max-width: 80%;
    }
    .badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .badge-appointment { background: rgba(59, 130, 246, 0.2); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-record { background: rgba(168, 85, 247, 0.2); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-history { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-search { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .section-header {
        font-size: 1.1rem; color: #10b981; font-weight: 600;
        margin-bottom: 10px; padding-bottom: 6px;
        border-bottom: 1px solid rgba(16, 185, 129, 0.2);
    }
    .log-entry {
        background: rgba(15, 22, 56, 0.8);
        border-left: 3px solid #10b981;
        padding: 10px 14px; margin: 6px 0;
        border-radius: 0 8px 8px 0;
        font-family: 'Cascadia Code', 'Fira Code', monospace;
        font-size: 0.82rem; color: #cbd5e1;
    }
    .log-success { border-left-color: #22c55e; }
    .log-failure { border-left-color: #ef4444; }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 10px 24px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "all_logs" not in st.session_state:
    st.session_state.all_logs = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🏥 HealthAgent AI")
    st.markdown("*Agentic Healthcare Assistant*")
    st.markdown("---")

    st.markdown("### How It Works")
    st.markdown("""
    ```
    Patient Query
         ↓
    🧠 Planner Agent
         ↓
    ┌────┼────┼────┐
    ↓    ↓    ↓    ↓
    📅   📋   🔍   🌐
    Book Update  RAG  Disease
    Appt Record History Search
    ```
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - **LLM:** GPT-4o-mini
    - **Framework:** LangGraph
    - **RAG:** FAISS + OpenAI Embeddings
    - **Database:** SQLite
    - **UI:** Streamlit
    """)

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    patients = get_all_patients()
    appts = get_all_appointments()
    doctors = get_all_doctors()
    st.metric("Patients", len(patients))
    st.metric("Doctors", len(doctors))
    st.metric("Appointments", len(appts))


# ── Main Title ──
st.markdown("""
<h1 style='text-align:center; background: linear-gradient(90deg, #10b981, #3b82f6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.4rem; margin-bottom: 0;'>
    🏥 HealthAgent AI
</h1>
<p style='text-align:center; color:#94a3b8; margin-top:4px; margin-bottom:24px;'>
    Agentic Healthcare Assistant for Medical Task Automation
</p>
""", unsafe_allow_html=True)

# ── Tabs ──
tab_chat, tab_patients, tab_appts, tab_history, tab_eval, tab_logs = st.tabs([
    "💬 Chat", "👥 Patients", "📅 Appointments", "📋 Medical History",
    "📊 Evaluation", "📝 Logs",
])


# ── TAB 1: CHAT ─────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown('<div class="section-header">Healthcare Assistant Chat</div>', unsafe_allow_html=True)

    user_message = st.text_input(
        "🩺 Your Request", value="",
        placeholder="e.g. Book a nephrologist for Ramesh Kulkarni...",
    )

    st.markdown("**Quick examples:**")
    ex_cols = st.columns(4)
    with ex_cols[0]:
        if st.button("📅 Book Appointment", use_container_width=True):
            st.session_state["_prefill"] = "Book a nephrologist appointment for Ramesh Kulkarni."
            st.rerun()
    with ex_cols[1]:
        if st.button("📋 Update Record", use_container_width=True):
            st.session_state["_prefill"] = "Update Anjali Mehra's record: she has moved to Mumbai."
            st.rerun()
    with ex_cols[2]:
        if st.button("🔍 Medical History", use_container_width=True):
            st.session_state["_prefill"] = "Summarize David Thompson's medical history."
            st.rerun()
    with ex_cols[3]:
        if st.button("🌐 Disease Info", use_container_width=True):
            st.session_state["_prefill"] = "What are the latest treatment methods for chronic kidney disease?"
            st.rerun()

    if "_prefill" in st.session_state:
        user_message = st.session_state.pop("_prefill")

    if st.button("🚀 Send", type="primary", use_container_width=True):
        if not user_message.strip():
            st.warning("Please enter a message.")
        else:
            with st.spinner("🧠 Agents processing your request..."):
                result = run_agent(user_message)

            intent = result.get("intent", "unknown")
            st.session_state.chat_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": user_message,
                "intent": intent,
                "response": result.get("response", ""),
            })
            st.session_state.all_logs.extend(result.get("logs", []))

            badge_class = {
                "appointment": "badge-appointment",
                "record_update": "badge-record",
                "history": "badge-history",
                "disease_search": "badge-search",
            }.get(intent, "badge-search")
            badge_label = intent.replace("_", " ").title()

            st.markdown(f'<span class="badge {badge_class}">{badge_label}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-bubble">🩺 {user_message}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-bubble">🏥 {result.get("response", "")}</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown('<div class="section-header">Conversation History</div>', unsafe_allow_html=True)
        for entry in reversed(st.session_state.chat_history):
            badge_class = {
                "appointment": "badge-appointment",
                "record_update": "badge-record",
                "history": "badge-history",
                "disease_search": "badge-search",
            }.get(entry["intent"], "badge-search")
            st.markdown(
                f'<span class="badge {badge_class}">{entry["intent"].replace("_"," ").title()}</span> '
                f'<small style="color:#64748b">{entry["timestamp"]}</small>',
                unsafe_allow_html=True)
            st.markdown(f'<div class="user-bubble">🩺 {entry["message"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-bubble">🏥 {entry["response"]}</div>', unsafe_allow_html=True)
            st.markdown("")


# ── TAB 2: PATIENTS ─────────────────────────────────────────────────────────
with tab_patients:
    st.markdown('<div class="section-header">Patient Database</div>', unsafe_allow_html=True)
    patients = get_all_patients()
    if patients:
        df = pd.DataFrame(patients)
        cols = st.columns(3)
        cols[0].markdown(
            f'<div class="kpi-card"><div class="kpi-value">{len(df)}</div>'
            f'<div class="kpi-label">Total Patients</div></div>', unsafe_allow_html=True)
        males = (df["gender"] == "Male").sum()
        females = (df["gender"] == "Female").sum()
        cols[1].markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#3b82f6">{males}</div>'
            f'<div class="kpi-label">Male</div></div>', unsafe_allow_html=True)
        cols[2].markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#ec4899">{females}</div>'
            f'<div class="kpi-label">Female</div></div>', unsafe_allow_html=True)
        st.markdown("")
        search = st.text_input("🔍 Search patients", placeholder="Type a name...")
        display_df = df[df["name"].str.contains(search, case=False)] if search else df
        st.dataframe(
            display_df.rename(columns={
                "patient_id": "ID", "phone": "Phone", "email": "Email",
                "name": "Name", "age": "Age", "gender": "Gender",
                "address": "Address", "summary": "Summary",
            }),
            use_container_width=True, hide_index=True)
    else:
        st.info("No patients in the database.")


# ── TAB 3: APPOINTMENTS ─────────────────────────────────────────────────────
with tab_appts:
    st.markdown('<div class="section-header">Appointments & Doctor Schedule</div>', unsafe_allow_html=True)

    # Doctor schedule
    st.markdown("#### 👨‍⚕️ Available Doctors")
    doctors = get_all_doctors()
    doc_df = pd.DataFrame(doctors)
    st.dataframe(
        doc_df.rename(columns={
            "doctor_id": "ID", "name": "Doctor", "specialty": "Specialty",
            "available_days": "Days", "available_hours": "Hours",
        }),
        use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📅 Booked Appointments")
    appts = get_all_appointments()
    if appts:
        appt_df = pd.DataFrame(appts)
        cols = st.columns(3)
        confirmed = (appt_df["status"] == "Confirmed").sum()
        cols[0].markdown(
            f'<div class="kpi-card"><div class="kpi-value">{len(appt_df)}</div>'
            f'<div class="kpi-label">Total</div></div>', unsafe_allow_html=True)
        cols[1].markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#22c55e">{confirmed}</div>'
            f'<div class="kpi-label">Confirmed</div></div>', unsafe_allow_html=True)
        cols[2].markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#f59e0b">{len(appt_df)-confirmed}</div>'
            f'<div class="kpi-label">Other</div></div>', unsafe_allow_html=True)
        st.markdown("")
        st.dataframe(
            appt_df.rename(columns={
                "appointment_id": "ID", "patient_name": "Patient",
                "doctor_name": "Doctor", "specialty": "Specialty",
                "date": "Date", "time_slot": "Time", "status": "Status",
                "created_at": "Created",
            }),
            use_container_width=True, hide_index=True)
    else:
        st.info("No appointments booked yet.")


# ── TAB 4: MEDICAL HISTORY (RAG) ────────────────────────────────────────────
with tab_history:
    st.markdown('<div class="section-header">Medical History Retrieval (RAG)</div>', unsafe_allow_html=True)
    st.markdown(
        "Search the FAISS vector store built from patient PDF reports and database records. "
        "The system uses RAG (Retrieval-Augmented Generation) to find and summarize relevant history."
    )

    hist_query = st.text_input(
        "🔍 Search medical history",
        placeholder="e.g. What is Anjali Mehra's diagnosis?",
    )
    if st.button("🔎 Search", type="primary", use_container_width=True):
        if hist_query.strip():
            with st.spinner("Searching medical records..."):
                context = retrieve_patient_context(hist_query)
                summary = _history_chain.invoke({
                    "user_message": hist_query,
                    "retrieved_context": context,
                })
            st.markdown("#### 📄 Retrieved Context")
            st.text(context)
            st.markdown("---")
            st.markdown("#### 🤖 AI Summary")
            st.markdown(summary)
        else:
            st.warning("Please enter a search query.")


# ── TAB 5: EVALUATION ───────────────────────────────────────────────────────
with tab_eval:
    st.markdown('<div class="section-header">Agent Evaluation Suite</div>', unsafe_allow_html=True)
    st.markdown(
        "Run 10 test cases across all 4 agent paths (appointment, records, history, disease search) "
        "to evaluate intent classification accuracy and response quality."
    )

    if st.button("▶️ Run Evaluation Suite", type="primary", use_container_width=True):
        with st.spinner("⏳ Running 10 test cases... This may take 60-90 seconds."):
            eval_df = run_evaluation()
            st.session_state.eval_results = eval_df

    if st.session_state.eval_results is not None:
        eval_df = st.session_state.eval_results
        metrics = compute_metrics(eval_df)

        m_cols = st.columns(4)
        m_cols[0].markdown(
            f'<div class="kpi-card"><div class="kpi-value">{metrics["total_tests"]}</div>'
            f'<div class="kpi-label">Total Tests</div></div>', unsafe_allow_html=True)
        m_cols[1].markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#22c55e">{metrics["correct"]}</div>'
            f'<div class="kpi-label">Correct</div></div>', unsafe_allow_html=True)
        m_cols[2].markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#a78bfa">{metrics["accuracy"]}</div>'
            f'<div class="kpi-label">Overall Accuracy</div></div>', unsafe_allow_html=True)
        m_cols[3].markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#f59e0b">{metrics["routing_accuracy"]}</div>'
            f'<div class="kpi-label">Routing Accuracy</div></div>', unsafe_allow_html=True)
        st.markdown("")

        def highlight_grade(val):
            if val == "CORRECT":
                return "background-color: rgba(34,197,94,0.15); color: #22c55e;"
            return "background-color: rgba(239,68,68,0.15); color: #ef4444;"

        styled = eval_df.style.applymap(highlight_grade, subset=["Grade"])
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ── TAB 6: LOGS ─────────────────────────────────────────────────────────────
with tab_logs:
    st.markdown('<div class="section-header">Agent Logs & Debugging</div>', unsafe_allow_html=True)

    if st.session_state.all_logs:
        total_actions = len(st.session_state.all_logs)
        successes = sum(1 for l in st.session_state.all_logs if l.get("status") == "success")
        failures = total_actions - successes

        log_cols = st.columns(3)
        log_cols[0].markdown(
            f'<div class="kpi-card"><div class="kpi-value">{total_actions}</div>'
            f'<div class="kpi-label">Total Actions</div></div>', unsafe_allow_html=True)
        log_cols[1].markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#22c55e">{successes}</div>'
            f'<div class="kpi-label">Successes</div></div>', unsafe_allow_html=True)
        log_cols[2].markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#ef4444">{failures}</div>'
            f'<div class="kpi-label">Failures</div></div>', unsafe_allow_html=True)
        st.markdown("")

        for log in reversed(st.session_state.all_logs):
            status_class = "log-success" if log.get("status") == "success" else "log-failure"
            status_icon = "✅" if log.get("status") == "success" else "❌"
            details = []
            if "action" in log:
                details.append(f"Action: {log['action']}")
            if "rag_chunks" in log:
                details.append(f"RAG Chunks: {log['rag_chunks']}")
            if "output" in log:
                details.append(f"Output: {log['output']}")

            st.markdown(
                f'<div class="log-entry {status_class}">'
                f'{status_icon} <b>[{log.get("agent", "Unknown")}]</b> '
                f'<small>{log.get("timestamp", "")[:19]}</small><br>'
                f'Input: {log.get("input", "N/A")}<br>'
                f'{"<br>".join(details)}'
                f'</div>', unsafe_allow_html=True)

        st.markdown("")
        if st.button("🗑️ Clear Logs", type="secondary"):
            st.session_state.all_logs = []
            st.rerun()
    else:
        st.info("No logs yet. Use the Chat tab to interact with agents.")


# ── Footer ──
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:0.8rem;">'
    '🏥 HealthAgent AI – Agentic Healthcare Assistant for Medical Task Automation | '
    'Built with LangGraph, GPT-4o-mini, FAISS & Streamlit'
    '</p>', unsafe_allow_html=True)
