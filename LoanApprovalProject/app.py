import streamlit as st
import pandas as pd
import joblib
from reportlab.pdfgen import canvas

model = joblib.load("loan_model.pkl")

def generate_pdf(result, reasons, applicant_data):
    pdf_file = "loan_report.pdf"
    c = canvas.Canvas(pdf_file)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, 800, "Loan Approval Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 760, f"Prediction Result: {result}")

    y = 720
    c.setFont("Helvetica-Bold", 13)
    c.drawString(100, y, "Applicant Details")

    c.setFont("Helvetica", 12)
    for key, value in applicant_data.items():
        y -= 22
        c.drawString(120, y, f"{key}: {value}")

    y -= 35
    c.setFont("Helvetica-Bold", 13)
    c.drawString(100, y, "Reasoning")

    c.setFont("Helvetica", 12)
    for reason in reasons:
        y -= 22
        c.drawString(120, y, f"- {reason}")

    c.save()
    return pdf_file


st.title("🏦 Loan Approval Prediction")
st.write("Machine Learning Based Loan Approval System")

st.divider()
Name = st.text_input("Applicant Name")
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.number_input("Dependents", min_value=0, max_value=5, value=0)
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Employment_Status = st.selectbox("Employment Status", ["Employed", "Self Employed"])

Applicant_Income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
Loan_Amount = st.number_input("Loan Amount", min_value=0.0, value=100.0)
Loan_Term = st.number_input("Loan Term", min_value=0.0, value=360.0)

Credit_History = st.selectbox("Credit History", ["Good", "Bad"])
Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
Age = st.number_input("Age", min_value=18, max_value=100, value=25)

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame([{
        "Gender": 1 if Gender == "Male" else 0,
        "Married": 1 if Married == "Yes" else 0,
        "Dependents": Dependents,
        "Education": 0 if Education == "Graduate" else 1,
        "Employment_Status": 0 if Employment_Status == "Employed" else 1,
        "Applicant_Income": Applicant_Income,
        "Coapplicant_Income": Coapplicant_Income,
        "Loan_Amount": Loan_Amount,
        "Loan_Term": Loan_Term,
        "Credit_History": 1 if Credit_History == "Good" else 0,
        "Property_Area": ["Rural", "Semiurban", "Urban"].index(Property_Area),
        "Age": Age
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 0:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    st.divider()

    if result == "Loan Approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    reasons = []

    if result == "Loan Approved":
        if Credit_History == "Good":
            reasons.append("Good credit history")

        if Applicant_Income >= 5000:
            reasons.append("Good applicant income")

        if Loan_Amount <= Applicant_Income:
            reasons.append("Loan amount is affordable")

        if Employment_Status == "Employed":
            reasons.append("Stable employment status")

    else:
        if Credit_History == "Bad":
            reasons.append("Poor credit history")

        if Applicant_Income < 5000:
            reasons.append("Low applicant income")

        if Loan_Amount > Applicant_Income:
            reasons.append("Loan amount is high compared to income")

        if Age < 21:
            reasons.append("Applicant age is low")

    if len(reasons) == 0:
        reasons.append("Prediction is based on overall applicant profile")

    st.subheader("Reasoning")

    for reason in reasons:
        st.write("-", reason)

    applicant_data = {
        "Name": Name,
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Employment Status": Employment_Status,
        "Applicant Income": Applicant_Income,
        "Coapplicant Income": Coapplicant_Income,
        "Loan Amount": Loan_Amount,
        "Loan Term": Loan_Term,
        "Credit History": Credit_History,
        "Property Area": Property_Area,
        "Age": Age
    }

    pdf_file = generate_pdf(result, reasons, applicant_data)

    with open(pdf_file, "rb") as file:
        st.download_button(
            label="📄 Download PDF Report",
            data=file,
            file_name="loan_report.pdf",
            mime="application/pdf"
        )