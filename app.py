import joblib
import pandas as pd
import streamlit as st

from features import add_features  # noqa: F401  (needed to load the saved pipeline)

st.set_page_config(page_title="Will This Visitor Buy?", page_icon="🛒", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


bundle = load_model()
model, THRESHOLD = bundle["model"], bundle["threshold"]

st.title("🛒 Will This Visitor Buy?")
st.write("Enter what a visitor did in their session and get the chance they will buy, "
         "plus what the website should do right now.")

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

with st.form("session"):
    st.subheader("Pages visited")
    c1, c2, c3 = st.columns(3)
    adm = c1.number_input("Account / admin pages", 0, 100, 2)
    inf = c2.number_input("Info pages", 0, 100, 0)
    prod = c3.number_input("Product pages", 0, 1000, 20)

    st.subheader("Time spent (minutes)")
    c1, c2, c3 = st.columns(3)
    adm_d = c1.number_input("On admin pages", 0.0, 600.0, 1.0)
    inf_d = c2.number_input("On info pages", 0.0, 600.0, 0.0)
    prod_d = c3.number_input("On product pages", 0.0, 1500.0, 12.0)

    st.subheader("Behaviour")
    c1, c2, c3 = st.columns(3)
    bounce = c1.slider("Bounce rate", 0.0, 0.2, 0.01, 0.005,
                       help="Share of visitors who leave right after landing on these pages")
    exit_r = c2.slider("Exit rate", 0.0, 0.2, 0.03, 0.005,
                       help="Share of page views that were the last one in a session")
    page_val = c3.number_input("Page value", 0.0, 400.0, 0.0,
                               help="Average value of the pages visited before a purchase (from Google Analytics)")

    st.subheader("Context")
    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("Month", MONTHS, index=MONTHS.index("Nov"))
    visitor = c2.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"],
                           format_func=lambda v: v.replace("_", " "))
    weekend = c3.checkbox("Weekend")
    special = st.slider("Closeness to a special day (1 = on the day)", 0.0, 1.0, 0.0, 0.2)

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "Administrative": adm, "Administrative_Duration": adm_d * 60,
        "Informational": inf, "Informational_Duration": inf_d * 60,
        "ProductRelated": prod, "ProductRelated_Duration": prod_d * 60,
        "BounceRates": bounce, "ExitRates": exit_r, "PageValues": page_val,
        "SpecialDay": special, "Month": month,
        # most common values in the data; not something a manager would know
        "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
        "VisitorType": visitor, "Weekend": weekend,
    }])
    try:
        p = float(model.predict_proba(row)[0, 1])
    except Exception as e:  # never crash in front of a user
        st.error(f"Could not score this session: {e}")
        st.stop()

    st.metric("Purchase probability", f"{p:.0%}")
    st.progress(p)

    if p >= THRESHOLD:
        st.success("**No action needed.** This visitor will most likely buy, so don't spend a discount on them.")
    elif p >= 0.30:
        st.warning("**Show a coupon.** The visitor is undecided, so a small discount could tip them into buying.")
    elif prod >= 20:
        st.info("**Open a chat.** They looked at many products but aren't converting; they may need help choosing.")
    else:
        st.info("**No action.** Looks like casual browsing.")

    with st.expander("How does this work?"):
        st.write(f"A {bundle['model_name']} model trained on 12,330 real sessions from an online store "
                 f"(UCI Online Shoppers Purchasing Intention dataset). It scored ROC-AUC 0.93 on unseen test data. "
                 f"Visitors above {THRESHOLD:.0%} are treated as likely buyers. "
                 "The biggest driver is Page value: visitors who reach pages that usually lead to a purchase are far more likely to buy.")

st.caption("Data: UCI Online Shoppers Purchasing Intention · https://archive.ics.uci.edu/dataset/468")
