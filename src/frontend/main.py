import requests
import streamlit as st

# Session State Setup
if "page" not in st.session_state:
    st.session_state.page = "main"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# Navigation: Seitenwechsel
def navigate_to(page):
    st.session_state.page = page
    st.rerun()


# Hauptseite
def main_page():
    st.title("Willkommen zur Gesundheits-App")
    st.subheader("Bitte anmelden oder registrieren")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            navigate_to("login")
    with col2:
        if st.button("Signup"):
            navigate_to("signup")


# Signup-Seite
def signup_page():
    st.title("Registrieren")

    username = st.text_input("Benutzername")
    first_name = st.text_input("Vorname")
    last_name = st.text_input("Nachname")
    email = st.text_input("Email")
    password = st.text_input("Passwort (mind. 6 Zeichen)", type="password")

    if st.button("Registrieren"):
        # Eingaben prüfen
        if not any([username, first_name, last_name, email]):
            st.warning("Bitte alle Felder ausfüllen.")
            return

        if len(password) < 6:
            st.warning("Passwort muss mindestens 6 Zeichen lang sein.")
            return

        payload = {
            "first_name": first_name,
            "last_name": last_name,
            "username": username,
            "email": email,
            "password": password,
        }

        try:
            response = requests.post(
                "http://backend:8000/api/v1/auth/signup", json=payload
            )

            if response.status_code == 201:
                st.success("Registrierung erfolgreich!")
                navigate_to("login")

            elif response.status_code == 403:
                detail = response.json().get("detail", "")
                if "email" in detail.lower():
                    st.warning("Ein Benutzer mit dieser E-Mail existiert bereits.")
                else:
                    st.error(f"Fehler 403: {detail}")

            else:
                st.error(f"Fehler: {response.status_code} – {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Netzwerkfehler: {e}")

    if st.button("Zurück"):
        navigate_to("main")


# Login-Seite
def login_page():
    st.title("Login")

    email = st.text_input("Email")
    password = st.text_input("Passwort", type="password")

    if st.button("Anmelden"):
        if not email or not password:
            st.warning("Bitte E-Mail und Passwort eingeben.")
            return

        if len(password) < 6:
            st.warning("Passwort muss mindestens 6 Zeichen lang sein.")
            return

        payload = {"email": email, "password": password}

        try:
            response = requests.post(
                "http://backend:8000/api/v1/auth/login", json=payload
            )

            if response.status_code == 200:
                data = response.json()

                # Tokens und User-Daten speichern
                st.session_state.logged_in = True
                st.session_state.access_token = data["access_token"]
                st.session_state.refresh_token = data["refresh_token"]
                st.session_state.username = data["user"]["username"]
                st.session_state.uid = data["user"]["uid"]

                navigate_to("prediction")

            elif response.status_code == 403:
                detail = response.json().get("detail", "")
                if "invalid" in detail.lower():
                    st.warning("Ungültige E-Mail oder Passwort.")
                else:
                    st.warning(f"Fehler 403: {detail}")

            elif response.status_code == 401:
                st.warning("Zugriff verweigert – nicht autorisiert.")

            else:
                st.error(f"Fehler {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Netzwerkfehler: {e}")

    if st.button("Zurück"):
        navigate_to("main")


# Prediction Page
def prediction_page():
    st.header(f"Willkommen {st.session_state.username}!")
    st.title("Herzkrankheit Vorhersage")
    st.write("Bitte geben Sie Ihre medizinischen Daten ein:")

    age = st.number_input("Alter", min_value=0)

    sex_label = st.selectbox("Geschlecht", ["Männlich (1)", "Weiblich (0)"])
    sex = 1 if sex_label.startswith("Männlich") else 0

    chest_pain_label = st.selectbox(
        "Brustschmerz-Typ",
        [
            "Typische Angina (1)",
            "Atypische Angina (2)",
            "Nicht-anginöse Schmerzen (3)",
            "Asymptomatisch (4)",
        ],
    )
    chest_pain_type = int(chest_pain_label.split("(")[-1][0])

    resting_bp_s = st.number_input("Ruheblutdruck (mm Hg)", min_value=0)

    cholesterol = st.number_input("Cholesterin (mg/dl)", min_value=0)

    fasting_label = st.selectbox(
        "Nüchternblutzucker > 120 mg/dl", ["Ja (1)", "Nein (0)"]
    )
    fasting_blood_sugar = 1 if fasting_label.startswith("Ja") else 0

    ecg_label = st.selectbox(
        "Ruhe-EKG Ergebnisse",
        [
            "Normal (0)",
            "ST-T-Wellen-Abnormalität (1)",
            "Linksventrikuläre Hypertrophie (2)",
        ],
    )
    resting_ecg = int(ecg_label.split("(")[-1][0])

    max_heart_rate = st.number_input("Maximale Herzfrequenz", min_value=0)

    angina_label = st.selectbox("Belastungsangina", ["Ja (1)", "Nein (0)"])
    exercise_angina = 1 if angina_label.startswith("Ja") else 0

    oldpeak = st.number_input("ST-Absenkung (Oldpeak)", format="%.2f")

    slope_label = st.selectbox(
        "ST-Strecken-Steigung", ["Aufsteigend (1)", "Flach (2)", "Absteigend (3)"]
    )
    st_slope = int(slope_label.split("(")[-1][0])

    if st.button("Vorhersagen"):
        # Check auf ungültige Werte (0 nicht erlaubt bei bestimmten Feldern)
        if age == 0:
            st.warning("Bitte geben Sie ein gültiges Alter ein (nicht 0).")
        elif resting_bp_s == 0:
            st.warning("Bitte geben Sie einen gültigen Ruheblutdruck ein (nicht 0).")
        elif cholesterol == 0:
            st.warning("Bitte geben Sie einen gültigen Cholesterinwert ein (nicht 0).")
        elif max_heart_rate == 0:
            st.warning(
                "Bitte geben Sie eine gültige maximale Herzfrequenz ein (nicht 0)."
            )
        elif oldpeak == 0:
            st.warning("Bitte geben Sie einen gültigen Oldpeak-Wert ein (nicht 0).")
        else:
            input_data = {
                "age": age,
                "sex": sex,
                "chest_pain_type": chest_pain_type,
                "resting_bp_s": resting_bp_s,
                "cholesterol": cholesterol,
                "fasting_blood_sugar": fasting_blood_sugar,
                "resting_ecg": resting_ecg,
                "max_heart_rate": max_heart_rate,
                "exercise_angina": exercise_angina,
                "oldpeak": oldpeak,
                "st_slope": st_slope,
            }

            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}

            try:
                response = requests.post(
                    "http://backend:8000/api/v1/predict/",
                    json=input_data,
                    headers=headers,
                )

                if response.status_code == 200:
                    result = response.json()
                    output = result["output"]
                    output_proba = result["output_proba"]

                    if output == 1:
                        st.error(
                            f"Herzkrankheit vorhergesagt, Wahrscheinlichkeit: {output_proba:.2f}"
                        )
                    else:
                        st.success(
                            f"Keine Herzkrankheit vorhergesagt, Wahrscheinlichkeit: {output_proba:.2f}"
                        )
                else:
                    st.error(f"Fehler {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Netzwerkfehler: {e}")

    if st.button("Logout"):
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}

        try:
            response = requests.post(
                "http://backend:8000/api/v1/auth/logout", headers=headers
            )
            if response.status_code == 200:
                st.success("Erfolgreich ausgeloggt.")
            else:
                st.warning(
                    f"Logout API Fehler: {response.status_code} - {response.text}"
                )
        except requests.exceptions.RequestException as e:
            st.error(f"Netzwerkfehler beim Logout: {e}")

        # Session-Daten löschen, egal ob API-Aufruf erfolgreich war oder nicht
        st.session_state.logged_in = False
        for key in ["access_token", "refresh_token", "username", "uid"]:
            if key in st.session_state:
                del st.session_state[key]
        navigate_to("main")


# Router
if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.logged_in:
    prediction_page()
else:
    navigate_to("main")
