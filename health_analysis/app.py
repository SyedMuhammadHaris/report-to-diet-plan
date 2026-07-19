import streamlit as st
from fpdf import FPDF

from llm_service import generate_diet_plan


def _diet_plan_to_pdf(diet_plan: str) -> bytes:
    safe_text = diet_plan.encode("latin-1", "ignore").decode("latin-1")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in safe_text.split("\n"):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 8, line, markdown=True)
    return bytes(pdf.output())

st.set_page_config(page_title="Health Report → Diet Plan", page_icon="🥗")

st.title("🥗 Health Report → Diet Plan")
st.caption("Upload a health report and get a personalized diet plan.")

with st.sidebar:
    st.header("Your Profile")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    weight_kg = st.number_input("Weight (kg)", min_value=1.0, value=70.0, step=0.5)
    height_cm = st.number_input("Height (cm)", min_value=1.0, value=170.0, step=0.5)
    activity_level = st.selectbox(
        "Activity level",
        ["Sedentary", "Lightly active", "Moderately active", "Very active"],
    )
    goal = st.selectbox(
        "Goal", ["Weight loss", "Weight gain", "Maintain weight", "Muscle gain", "General health"]
    )
    dietary_restrictions = st.text_input(
        "Dietary restrictions (e.g. vegetarian, vegan)", ""
    )
    allergies = st.text_input("Allergies (comma separated)", "")

uploaded_file = st.file_uploader(
    "Upload health report", type=["pdf", "png", "jpg", "jpeg", "txt", "csv"]
)

generate_clicked = st.button("Generate Diet Plan", type="primary", disabled=uploaded_file is None)

if generate_clicked and uploaded_file is not None:
    profile = {
        "age": age,
        "gender": gender,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "activity_level": activity_level,
        "dietary_restrictions": dietary_restrictions,
        "allergies": allergies,
        "goal": goal,
    }

    with st.spinner("Analyzing report and generating diet plan..."):
        try:
            diet_plan = generate_diet_plan(uploaded_file, profile)
            st.subheader("Your Diet Plan")
            st.markdown(diet_plan)
            st.download_button(
                "Download Diet Plan as PDF",
                data=_diet_plan_to_pdf(diet_plan),
                file_name="diet_plan.pdf",
                mime="application/pdf",
            )
        except NotImplementedError as e:
            st.info(f"LLM call layer not wired up yet: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
