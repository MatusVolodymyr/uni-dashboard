"""Demo access control.

A simple role picker that scopes which data a user can see. This is a stand-in:
once the dashboard is embedded in the university portal, the role (and the faculty
a dean belongs to) will come from the platform's authentication instead of a
dropdown. The rest of the app only needs the scoped dataframe and the role.
"""
import pandas as pd
import streamlit as st

ROLE_RECTOR = "Ректор / керівництво університету"
ROLE_DEAN = "Декан факультету"
ROLES = [ROLE_RECTOR, ROLE_DEAN]


def access_control(df: pd.DataFrame):
    """Render the role picker in the sidebar and return (scoped_df, role, faculty).

    Rector: full access to all faculties.
    Dean:   data restricted to one chosen faculty, across every page.
    The selection persists across pages via st.session_state (keys role / dean_faculty).
    """
    st.sidebar.header("🔐 Рівень доступу")
    role = st.sidebar.selectbox(
        "Роль",
        ROLES,
        key="role",
        help="Демо-перемикач. Після інтеграції з порталом роль і доступ "
             "визначатимуться автоматично за автентифікацією користувача.",
    )

    scope_faculty = None
    if role == ROLE_DEAN:
        faculties = sorted(df["faculty"].unique())
        scope_faculty = st.sidebar.selectbox("Ваш факультет", faculties, key="dean_faculty")
        df = df[df["faculty"] == scope_faculty]
        st.sidebar.success(f"Доступ обмежено факультетом:\n\n**{scope_faculty}**")
    else:
        st.sidebar.info("Доступ до всіх факультетів університету.")

    st.sidebar.divider()
    return df, role, scope_faculty
