import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

st.set_page_config(page_title="AI Spreadsheet Cleaner", layout="wide")

# ------------------------- Helpers -------------------------

def read_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None


def download_df_as_csv(df, filename="cleaned.csv"):
    towrite = io.BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return towrite


def guess_column_type(series: pd.Series):
    """Return a guessed, simple type for a column: date/number/text/email/phone"""
    s = series.dropna().astype(str)
    if s.empty:
        return 'empty'
    # date detection
    date_like = 0
    num_like = 0
    email_like = 0
    phone_like = 0
    for v in s.sample(min(len(s), 50)):
        v = v.strip()
        # number
        try:
            float(v.replace(',', ''))
            num_like += 1
        except:
            pass
        # date
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y"):
            try:
                datetime.strptime(v, fmt)
                date_like += 1
                break
            except:
                pass
        # email
        if "@" in v and "." in v.split('@')[-1]:
            email_like += 1
        # phone simple heuristic
        digits = ''.join(ch for ch in v if ch.isdigit())
        if 7 <= len(digits) <= 15:
            phone_like += 1
    counts = {"date": date_like, "number": num_like, "email": email_like, "phone": phone_like}
    return max(counts, key=counts.get)


def standardize_strings(s: pd.Series, case: str = 'title'):
    if case == 'lower':
        return s.astype(str).str.strip().str.lower()
    if case == 'upper':
        return s.astype(str).str.strip().str.upper()
    if case == 'title':
        return s.astype(str).str.strip().str.title()
    return s.astype(str).str.strip()


def fix_dates(s: pd.Series):
    # try to parse dates robustly
    return pd.to_datetime(s, errors='coerce')


def numeric_clean(s: pd.Series):
    def parse_num(v):
        if pd.isna(v):
            return np.nan
        try:
            v = str(v).replace(',', '').strip()
            return float(v)
        except:
            return np.nan
    return s.apply(parse_num)


# ------------------------- UI -------------------------

st.title("AI Spreadsheet Cleaner — Starter (Streamlit)")
st.markdown(
    "Upload a CSV/XLSX file and apply common cleaning operations. This starter app focuses on robust heuristics and manual controls so you can quickly fix messy sheets and export a clean CSV."
)

with st.sidebar:
    st.header("Options")
    sample_rows = st.number_input("Preview rows", min_value=5, max_value=200, value=25)
    auto_guess = st.checkbox("Auto-guess column types & suggest fixes", value=True)
    drop_empty_cols = st.checkbox("Drop fully empty columns", value=True)
    drop_duplicates = st.checkbox("Remove duplicate rows (exact match)", value=True)
    dedupe_subset = st.text_input("Dedupe subset columns (comma-separated, optional)")
    fill_na_method = st.selectbox("Fill missing values with", ['None', 'Empty string', '0', 'Forward fill', 'Custom value'])
    custom_fill_value = st.text_input("Custom fill value (used if selected)")
    apply_all = st.button("Apply selected cleaning")

uploaded = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xls', 'xlsx'])

if uploaded is not None:
    df = read_file(uploaded)
    if df is None:
        st.stop()

    st.subheader("Preview — Raw Data")
    st.dataframe(df.head(sample_rows))

    st.markdown("---")

    # automatic suggestions
    col_meta = {}
    for c in df.columns:
        guessed = guess_column_type(df[c])
        col_meta[c] = guessed

    if auto_guess:
        st.subheader("Auto-Detected Column Types & Quick Actions")
        cols = st.columns(3)
        i = 0
        user_actions = {}
        for c, guess in col_meta.items():
            with cols[i % 3]:
                st.markdown(f"**{c}**")
                st.caption(f"Guessed: {guess}")
                # present possible quick fixes
                if guess == 'date':
                    user_actions[c] = st.checkbox(f"Parse dates for {c}", value=True, key=f"date_{c}")
                elif guess == 'number':
                    user_actions[c] = st.checkbox(f"Clean numbers for {c}", value=True, key=f"num_{c}")
                elif guess == 'email':
                    user_actions[c] = st.checkbox(f"Trim/lowercase emails for {c}", value=True, key=f"email_{c}")
                elif guess == 'phone':
                    user_actions[c] = st.checkbox(f"Normalize phone for {c}", value=False, key=f"phone_{c}")
                else:
                    # generic actions for text
                    user_actions[c] = st.selectbox(f"Text case for {c}", ['none', 'strip', 'lower', 'upper', 'title'], index=1, key=f"case_{c}")
            i += 1

    st.markdown("---")

    st.subheader("Manual Column Operations")
    st.markdown("Select a column to inspect and apply manual operations.")
    chosen_col = st.selectbox("Choose column", options=[None] + list(df.columns))

    if chosen_col:
        col = df[chosen_col]
        st.write(col.head(sample_rows))
        st.write(f"Detected type: {col_meta.get(chosen_col)}")
        st.markdown("**Manual Transformations**")
        do_strip = st.checkbox("Strip whitespace", value=True, key=f"strip_{chosen_col}")
        case = st.selectbox("Change case", ['none', 'lower', 'upper', 'title'], key=f"case2_{chosen_col}")
        parse_date = st.checkbox("Parse as date", key=f"pdate_{chosen_col}")
        parse_number = st.checkbox("Parse as number", key=f"pnum_{chosen_col}")
        replace_map_text = st.text_area("Replace text map (one per line, old=>new)")

        if st.button("Apply to column", key=f"applycol_{chosen_col}"):
            if do_strip:
                df[chosen_col] = df[chosen_col].astype(str).str.strip()
            if case != 'none':
                df[chosen_col] = standardize_strings(df[chosen_col], case=case)
            if parse_date:
                df[chosen_col] = fix_dates(df[chosen_col])
            if parse_number:
                df[chosen_col] = numeric_clean(df[chosen_col])
            if replace_map_text.strip():
                for line in replace_map_text.splitlines():
                    if '=>' in line:
                        old, new = line.split('=>', 1)
                        df[chosen_col] = df[chosen_col].astype(str).str.replace(old.strip(), new.strip(), regex=False)
            st.success(f"Applied transformations to {chosen_col}")
            st.experimental_rerun()

    st.markdown("---")

    st.subheader("Batch Preview of Planned Changes")
    preview_df = df.copy()

    # automatic actions performed in-memory for preview only
    # apply auto suggestions from earlier
    if auto_guess:
        for c, guess in col_meta.items():
            key = None
            if guess == 'date':
                key = f"date_{c}"
                if st.session_state.get(key, False):
                    preview_df[c] = fix_dates(preview_df[c])
            elif guess == 'number':
                key = f"num_{c}"
                if st.session_state.get(key, False):
                    preview_df[c] = numeric_clean(preview_df[c])
            elif guess == 'email':
                key = f"email_{c}"
                if st.session_state.get(key, False):
                    preview_df[c] = standardize_strings(preview_df[c], case='lower')
            elif guess == 'phone':
                key = f"phone_{c}"
                if st.session_state.get(key, False):
                    preview_df[c] = preview_df[c].astype(str).str.replace(r'[^0-9+]', '', regex=True)
            else:
                # text case selection
                key = f"case_{c}"
                val = st.session_state.get(key, 'strip')
                if val in ('lower', 'upper', 'title'):
                    preview_df[c] = standardize_strings(preview_df[c], case=val)
                elif val == 'strip':
                    preview_df[c] = preview_df[c].astype(str).str.strip()

    # manual global fills
    if fill_na_method != 'None':
        if fill_na_method == 'Empty string':
            preview_df = preview_df.fillna('')
        elif fill_na_method == '0':
            preview_df = preview_df.fillna(0)
        elif fill_na_method == 'Forward fill':
            preview_df = preview_df.fillna(method='ffill')
        elif fill_na_method == 'Custom value':
            preview_df = preview_df.fillna(custom_fill_value)

    if drop_empty_cols:
        preview_df = preview_df.dropna(axis=1, how='all')

    if drop_duplicates:
        if dedupe_subset.strip():
            subset_cols = [c.strip() for c in dedupe_subset.split(',') if c.strip() in preview_df.columns]
            if subset_cols:
                preview_df = preview_df.drop_duplicates(subset=subset_cols)
            else:
                preview_df = preview_df.drop_duplicates()
        else:
            preview_df = preview_df.drop_duplicates()

    st.write(preview_df.head(sample_rows))

    st.markdown("---")

    st.subheader("Summary & Export")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown("**Original shape**")
        st.write(df.shape)
        st.markdown("**Preview shape after planned cleaning**")
        st.write(preview_df.shape)

    with col2:
        st.markdown("**Column summary (sample)**")
        summary = preview_df.describe(include='all').transpose()
        st.dataframe(summary)

    with col3:
        st.markdown("**Export**")
        buf = download_df_as_csv(preview_df)
        st.download_button(label="Download cleaned CSV", data=buf, file_name="cleaned.csv", mime='text/csv')

    st.markdown("---")
    st.info("This starter app uses heuristics to suggest fixes. For production, you can add OpenAI-powered suggestions for header mapping, fuzzy deduplication, and automated rule generation. If you want that, tell me and I will add examples that call an LLM using your API key.")

else:
    st.info("Upload a CSV or Excel file to begin. Want a version that connects to OpenAI for smart suggestions? Say 'add AI' and I'll add it.")
