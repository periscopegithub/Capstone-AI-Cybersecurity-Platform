# helpers.py
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
import pytz


def load_questions():
    return pd.read_excel("Questionnaire.xlsx")


def load_industries():
    with open("texts/industries.txt") as f:
        industries = [line.strip() for line in f.readlines()]
    return industries


def load_countries():
    with open("texts/countries.txt") as f:
        countries = [line.strip().upper() for line in f.readlines()]
    countries.sort()
    return countries


def load_invitation_codes():
    with open("texts/invitation_codes.txt") as f:
        invitation_codes = [line.strip() for line in f.readlines()]
    return invitation_codes


def load_text(file_name):
    TEXT_FOLDER = "texts"
    with open(os.path.join(TEXT_FOLDER, file_name)) as f:
        return f.read()


def get_current_datetime():
    local_tz = pytz.timezone("Asia/Hong_Kong")  # Adjust the timezone as needed
    now_local = datetime.now(local_tz)
    current_time_str = now_local.strftime("%H:%M:%S")
    return f"{current_time_str} today"


def get_num_respondents():
    responses_file = "Responses/all_responses.csv"
    if os.path.exists(responses_file):
        df = pd.read_csv(responses_file)
        return df["Response ID"].max()
    return 0


def get_index_score():
    responses_file = "Responses/all_responses.csv"
    if os.path.exists(responses_file):
        df = pd.read_csv(responses_file)
        mean_score = df["Overall Score"].mean()
        if pd.notna(mean_score):  # Check if mean_score is not NaN
            return round(mean_score * 20, 1)
    return 0


def get_cyber_threat_data():
    df = pd.read_csv("dexpose_io.csv")
    df["Date (UTC)"] = pd.to_datetime(df["Date (UTC)"])
    last_90_days = datetime.now(timezone.utc) - timedelta(days=90)
    recent_data = df[df["Date (UTC)"] >= last_90_days]
    num_ransom_attacks = len(recent_data[recent_data["Alert Type"] == "Ransomware"])
    num_data_breaches = len(recent_data[recent_data["Alert Type"] == "Data Breach"])

    top_ransomware_counts = (
        recent_data[recent_data["Alert Type"] == "Ransomware"]["Ransomware"]
        .value_counts()
        .head(3)
    )
    top_ransomware = top_ransomware_counts.index.tolist()
    top_ransomware_counts = top_ransomware_counts.tolist()

    return num_ransom_attacks, num_data_breaches, top_ransomware, top_ransomware_counts


def save_responses(
    response_time,
    industry,
    country,
    responses,
    category_averages,
    answers,
    questions_df,
):
    responses_folder = "Responses"
    responses_file = os.path.join(responses_folder, "all_responses.csv")

    if not os.path.exists(responses_folder):
        os.makedirs(responses_folder)

    if os.path.exists(responses_file):
        all_responses = pd.read_csv(responses_file)
        next_response_id = all_responses["Response ID"].max() + 1
    else:
        next_response_id = 1
        all_responses = pd.DataFrame()

    overall_score = (
        sum(category_averages.values()) / len(category_averages)
        if category_averages
        else 0
    )

    response_data = {
        "Response ID": next_response_id,
        "Response Time (UTC)": response_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Industry": industry,
        "Country/Region": country,
        **{
            f'Question {row["Question ID"]}': responses[i]
            for i, row in questions_df.iterrows()
        },
        **{f"{cat}": avg for cat, avg in category_averages.items()},
        "Overall Score": overall_score,
    }

    new_response_df = pd.DataFrame([response_data])
    all_responses = pd.concat([all_responses, new_response_df], ignore_index=True)
    all_responses.to_csv(responses_file, index=False)

    answers_df = pd.DataFrame(answers)
    answers_file_path = os.path.join(
        responses_folder, f"Respondent_{next_response_id}.csv"
    )
    answers_df.to_csv(answers_file_path, index=False)
    print(f"Respondent's responses have been saved to {answers_file_path}")

    index_score = all_responses["Overall Score"].mean()
    std_dev = all_responses["Overall Score"].std()

    return (
        all_responses,
        answers_df,
        next_response_id,
        overall_score,
        index_score,
        std_dev,
    )


def detailed_comparison_with_actual_percentiles(
    response_df, category_averages, respondent_id
):
    results = {}

    respondent_row = response_df[response_df["Response ID"] == respondent_id]
    if respondent_row.empty:
        return "No data found for respondent ID {}".format(respondent_id)

    score_columns = [f"{cat}" for cat in category_averages.keys()]

    for column in score_columns:
        if column in response_df.columns:
            all_scores = response_df[column].dropna()
            total_respondents = len(all_scores)
            respondent_score = respondent_row.iloc[0][column]
            mean = all_scores.mean()
            std = all_scores.std()

            sorted_scores = all_scores.sort_values().reset_index(drop=True)
            rank = sorted_scores[sorted_scores == respondent_score].index[0]
            percentile = (rank / len(sorted_scores)) * 100
            rounded_percentile = round(percentile)

            if std > 0:
                z_score = (respondent_score - mean) / std
            else:
                z_score = 0

            direction = "above" if z_score >= 0 else "below"

            message = (
                f"{column}: "
                f"Your scored {respondent_score * 20:.1f}, {direction} the average score of {mean * 20:.1f}. "
                f"Among {total_respondents} respondents, you are at the {rounded_percentile}th percentile, and {abs(z_score):.1f} S.D. {direction} the mean."
            )

            results[column] = message
        else:
            results[column] = "Data not available for this category."

    return results


def detailed_comparison_with_industry_percentiles(
    response_df, category_averages, respondent_id, respondent_industry
):
    results = {}

    respondent_row = response_df[response_df["Response ID"] == respondent_id]
    if respondent_row.empty:
        return "No data found for respondent ID {}".format(respondent_id)

    industry_data = response_df[response_df["Industry"] == respondent_industry]
    score_columns = [f"{cat}" for cat in category_averages.keys()]

    for column in score_columns:
        if column in industry_data.columns:
            industry_scores = industry_data[column].dropna()
            total_respondents = len(industry_scores)
            respondent_score = respondent_row.iloc[0][column]
            mean = industry_scores.mean()
            std = industry_scores.std()

            sorted_scores = industry_scores.sort_values().reset_index(drop=True)
            rank = sorted_scores[sorted_scores == respondent_score].index[0]
            percentile = (rank / len(sorted_scores)) * 100
            rounded_percentile = round(percentile)

            if std > 0:
                z_score = (respondent_score - mean) / std
            else:
                z_score = 0

            direction = "above" if z_score >= 0 else "below"

            message = (
                f"{column}: "
                f"Your scored {respondent_score * 20:.1f}, {direction} the industry's average of {mean * 20:.1f}. "
                f"Among {total_respondents} industry peers, you are at the {rounded_percentile}th percentile, and {abs(z_score):.1f} S.D. {direction} the mean."
            )

            results[column] = message
        else:
            results[column] = "Data not available for this category."

    return results


def load_valid_emails():
    with open("texts/emails.txt") as f:
        valid_emails = [line.strip() for line in f.readlines()]
    return valid_emails


def is_valid_email(email, valid_emails):
    return "@" in email and email in valid_emails


def get_industry_average_score(industry):
    responses_file = "Responses/all_responses.csv"
    if os.path.exists(responses_file):
        df = pd.read_csv(responses_file)
        industry_responses = df[df["Industry"] == industry]
        mean_score = industry_responses["Overall Score"].mean()
        if pd.notna(mean_score):  # Check if mean_score is not NaN
            return round(mean_score * 20, 1)
    return 0


def get_first_response_datetime():
    responses_file = "Responses/all_responses.csv"
    if os.path.exists(responses_file):
        df = pd.read_csv(responses_file)
        if not df.empty and "Response Time (UTC)" in df.columns:
            # Try multiple formats
            datetime_formats = [
                "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M",
            ]
            first_response_datetime_utc = None

            for fmt in datetime_formats:
                try:
                    first_response_datetime_utc = pd.to_datetime(
                        df["Response Time (UTC)"], format=fmt, utc=True
                    ).min()
                    break  # Exit the loop if successful
                except ValueError:
                    continue  # Try the next format

            if first_response_datetime_utc is not None and pd.notna(
                first_response_datetime_utc
            ):
                # Convert to local time zone (example: 'Asia/Hong_Kong')
                local_tz = pytz.timezone(
                    "Asia/Hong_Kong"
                )  # Adjust the timezone as needed
                first_response_datetime_local = first_response_datetime_utc.tz_convert(
                    local_tz
                )
                return first_response_datetime_local.strftime("%Y-%m-%d")
    # Return today's date in local time zone if no entry or file not found
    today_local = datetime.now(pytz.timezone("Asia/Hong_Kong")).strftime("%Y-%m-%d")
    return today_local


def calculate_industry_scores():
    responses_df = pd.read_csv('Responses/all_responses.csv')
    questions_df = pd.read_excel('Questionnaire.xlsx')

    question_categories = questions_df['Question Category'].unique().tolist()

    industry_scores = {}

    industries = responses_df['Industry'].unique()
    for industry in industries:
        industry_data = responses_df[responses_df['Industry'] == industry]
        overall_score = industry_data['Overall Score'].mean()
        category_scores = {}
        for category in question_categories:
            category_scores[category] = industry_data[category].mean()
        industry_scores[industry] = {'Overall Score': overall_score, **category_scores}

    return industry_scores, question_categories