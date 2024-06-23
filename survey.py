from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    session,
    redirect,
    url_for,
    flash,
)

# from celery_app import celery
from celery.exceptions import OperationalError
from celery.utils.log import get_task_logger
from tasks import generate_report_async
from datetime import datetime, timezone
from collections import defaultdict
from report_generation import generate_detailed_response_report
from helpers import (
    load_questions,
    load_industries,
    load_countries,
    save_responses,
    detailed_comparison_with_actual_percentiles,
    detailed_comparison_with_industry_percentiles,
    load_text,
    get_current_datetime,
    get_num_respondents,
    get_index_score,
    get_cyber_threat_data,
    load_invitation_codes,
    get_industry_average_score,
    get_first_response_datetime,
)
from celery.utils.log import get_task_logger
import os
import url_classifier
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd


# Set up a logger specifically for Celery tasks
logger = get_task_logger(__name__)

app = Flask(__name__)
app.secret_key = "capstone"


def load_email_model():
    # checkpoint_path = r"C:\Users\Karl\Documents\Projects\llm-email-spam-detection\experiments\checkpoint-26835"
    # model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "periscopehf/capstone-email-classifier"
    )
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    return model, tokenizer


email_model, email_tokenizer = load_email_model()


@app.route("/survey", methods=["GET", "POST"])
def survey():
    if request.method == "POST":
        invitation_code = request.form.get("invitation_code")
        valid_codes = load_invitation_codes()

        if invitation_code and invitation_code not in valid_codes:
            flash("Invalid invitation code. Please try again.")
            return redirect(url_for("welcome"))

        session["invitation_code"] = invitation_code
        came_from_welcome = True
    else:
        came_from_welcome = False

    questions_df = load_questions()
    industries = load_industries()
    countries = load_countries()

    return render_template(
        "survey.html",
        questions_df=questions_df.to_dict(orient="records"),
        industries=industries,
        countries=countries,
        came_from_welcome=came_from_welcome,
    )


@app.route("/welcome")
def welcome():
    platform_introduction = load_text("platform_introduction.txt")
    index_introduction = load_text("index_introduction.txt")
    ai_assessment_introduction = load_text("ai_assessment_introduction.txt")
    country_industry_report_introduction = load_text(
        "country_industry_report_introduction.txt"
    )
    cyber_threat_trends_introduction = load_text("cyber_threat_trends_introduction.txt")
    phishing_prevention_introduction = load_text("phishing_prevention_introduction.txt")
    suspicious_email_detector_introduction = load_text(
        "suspicious_email_detector_introduction.txt"
    )
    suspicious_website_detector_introduction = load_text(
        "suspicious_website_detector_introduction.txt"
    )

    current_date_time = get_current_datetime()
    num_respondents = get_num_respondents()
    index_score = get_index_score()
    first_response_date = get_first_response_datetime()

    num_ransom_attacks, num_data_breaches, top_ransomware, top_ransomware_counts = (
        get_cyber_threat_data()
    )
    top_ransomware_1, top_ransomware_2, top_ransomware_3 = top_ransomware + [None] * (
        3 - len(top_ransomware)
    )
    top_ransomware_count_1, top_ransomware_count_2, top_ransomware_count_3 = (
        top_ransomware_counts + [None] * (3 - len(top_ransomware_counts))
    )

    countries = load_countries()
    industries = load_industries()

    # Load all responses data and calculate average scores by industry and category
    all_responses = pd.read_csv("Responses/all_responses.csv")
    questions_df = load_questions()
    question_categories = questions_df["Question Category"].unique().tolist()

    # Calculate number of respondents for each industry
    industry_counts = all_responses["Industry"].value_counts().to_dict()

    industry_avg_scores = (
        all_responses.groupby("Industry")
        .mean(numeric_only=True)
        .round(1)
        .dropna(how="all")
    )

    # Add the number of respondents to the average scores
    industry_avg_scores["No. of Respondents"] = industry_avg_scores.index.map(
        industry_counts
    )

    # Prepare tables with max 4 columns each
    max_columns = 4
    industry_tables = []
    for i in range(0, len(question_categories), max_columns):
        categories_subset = question_categories[i : i + max_columns]
        subset_data = {
            "Industry": industry_avg_scores.index.tolist(),
            "No. of Respondents": industry_avg_scores["No. of Respondents"].tolist(),
            "Overall Score": industry_avg_scores["Overall Score"].tolist(),
        }
        for category in categories_subset:
            if category in industry_avg_scores.columns:
                subset_data[category] = industry_avg_scores[category].tolist()
            else:
                subset_data[category] = [None] * len(industry_avg_scores)

        industry_tables.append(
            {
                "question_categories": ["Overall Score"] + categories_subset,
                "industry_scores": {
                    industry: {cat: subset_data[cat][j] for cat in subset_data}
                    for j, industry in enumerate(subset_data["Industry"])
                },
            }
        )

    # Calculate overall performance
    overall_performance = {
        "Industry": "Overall Performance",
        "No. of Respondents": all_responses.shape[0],
        "Overall Score": all_responses["Overall Score"].mean().round(1),
    }
    for category in question_categories:
        if category in all_responses.columns:
            overall_performance[category] = (
                all_responses[category].mean(numeric_only=True).round(1)
            )

    return render_template(
        "welcome.html",
        platform_introduction=platform_introduction,
        index_introduction=index_introduction,
        ai_assessment_introduction=ai_assessment_introduction,
        country_industry_report_introduction=country_industry_report_introduction,
        cyber_threat_trends_introduction=cyber_threat_trends_introduction,
        phishing_prevention_introduction=phishing_prevention_introduction,
        suspicious_email_detector_introduction=suspicious_email_detector_introduction,
        suspicious_website_detector_introduction=suspicious_website_detector_introduction,
        current_date_time=current_date_time,
        num_respondents=num_respondents,
        index_score=index_score,
        first_response_date=first_response_date,
        num_ransom_attacks=num_ransom_attacks,
        num_data_breaches=num_data_breaches,
        top_ransomware_1=top_ransomware_1,
        top_ransomware_2=top_ransomware_2,
        top_ransomware_3=top_ransomware_3,
        top_ransomware_count_1=top_ransomware_count_1,
        top_ransomware_count_2=top_ransomware_count_2,
        top_ransomware_count_3=top_ransomware_count_3,
        countries=countries,
        industries=industries,
        industry_tables=industry_tables,
        overall_performance=overall_performance,
    )


@app.route("/check_email", methods=["POST"])
def check_email():
    data = request.get_json()
    email_text = data.get("email", "").strip()
    if not email_text:
        return (
            jsonify(
                {"message": "No email text provided. Please paste an email message."}
            ),
            400,
        )

    classification = classify_email(email_text, email_model, email_tokenizer)

    if classification == "spam":
        return jsonify(
            {
                "message": "This email looks suspicious. Avoid clicking on any links or downloading any attachments.",
            }
        )
    elif classification == "ham":
        return jsonify(
            {
                "message": "This email may be safe, but always exercise caution clicking on any links or downloading any attachments."
            }
        )

    return jsonify({"message": classification})


def classify_email(text, model, tokenizer):
    prefix = "classify as ham or spam: "
    encoded_input = tokenizer(
        prefix + text, return_tensors="pt", truncation=True, max_length=512
    )
    output = model.generate(**encoded_input, max_length=5)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.strip()


@app.route("/submit_report_request", methods=["POST"])
def submit_report_request():
    country = request.form.get("country")
    industry = request.form.get("industry")
    if not industry or not country:
        return "Industry and country selection are required.", 400

    # Start background tasks
    try:
        task = generate_report_async.apply_async(
            args=[None, None, country, industry, None, None]  # Pass default None values
        )
        return render_template(
            "thank_you.html",
            task_id=task.id,
            disclaimer=load_text("disclaimer.txt"),
            country=country,
            from_welcome=True,
        )
    except OperationalError as e:
        return "Failed to connect to the task queue. Please try again later.", 500


@app.route("/submit", methods=["POST"])
def submit():
    email = session.get("email")

    questions_df = load_questions()
    responses = []
    answers = []
    response_time = datetime.now(timezone.utc)
    category_averages = {}

    industry = request.form.get("industry")
    country = request.form.get("country")
    if not industry or not country:
        return "Industry and country selection are required.", 400

    for index, row in questions_df.iterrows():
        score = request.form.get(f"score_{row['Question ID']}")
        if score:
            responses.append(int(score))
            answers.append(
                {
                    "Question ID": row["Question ID"],
                    "Category": row["Question Category"],
                    "Subcategory": row["Subcategory"],
                    "Question Text": row["Question Text"],
                    "Score": score,
                    "Score Description": row[f"Score {score}"],
                }
            )

    category_scores = defaultdict(list)
    for answer in answers:
        category_scores[answer["Category"]].append(int(answer["Score"]))
    category_averages = {
        cat: sum(scores) / len(scores) if scores else 0
        for cat, scores in category_scores.items()
    }

    all_responses, answers_df, next_response_id, overall_score, index_score, std_dev = (
        save_responses(
            response_time,
            industry,
            country,
            responses,
            category_averages,
            answers,
            questions_df,
        )
    )

    if std_dev > 0:
        z_score = (overall_score - index_score) / std_dev
    else:
        z_score = 0
    direction = "above" if z_score >= 0 else "below"

    overall_score_message = (
        f"You scored {overall_score * 20:.1f} in this survey. "
        f"You are {abs(z_score):.2f} S.D. {direction} the index score of {index_score * 20:.1f}."
    )

    detailed_results = detailed_comparison_with_actual_percentiles(
        all_responses, category_averages, next_response_id
    )

    # Calculate industry average score
    industry_average_score = get_industry_average_score(industry)
    if industry_average_score > 0:
        direction = (
            "above" if industry_average_score * 20 > industry_average_score else "below"
        )
        industry_score_message = (
            f"Your overall score is {overall_score * 20:.1f}, "
            f"{direction} your industry's average overall score of {industry_average_score:.1f}."
        )
    else:
        industry_score_message = (
            f"Your overall score is {overall_score * 20:.1f}. "
            f"There is not enough data to calculate the average overall score for your industry."
        )

    industry_detailed_results = detailed_comparison_with_industry_percentiles(
        all_responses, category_averages, next_response_id, industry
    )

    response_reports = generate_detailed_response_report(questions_df, answers_df)

    # Log the arguments
    logger.info(
        f"Arguments: response_reports={response_reports}, category_averages={category_averages}, country={country}, industry={industry}"
    )

    # Start background tasks
    try:
        task = generate_report_async.apply_async(
            args=[
                response_reports,
                category_averages,
                country,
                industry,
                overall_score_message,
                detailed_results,
            ]
        )

        return render_template(
            "thank_you.html",
            overall_score_message=overall_score_message,
            detailed_results=detailed_results,
            industry_score_message=industry_score_message,  # Pass the new message to the template
            industry_detailed_results=industry_detailed_results,  # Pass the new detailed results to the template
            task_id=task.id,  # Pass the task id to the template
            disclaimer=load_text("disclaimer.txt"),
            country=country,  # Pass the country to the template
        )
    except OperationalError as e:
        return "Failed to connect to the task queue. Please try again later.", 500


@app.route("/check_task/<task_id>")
def check_task(task_id):
    task = generate_report_async.AsyncResult(task_id)
    if task.state == "SUCCESS":
        response = task.result
        response.update({"status": "SUCCESS"})
        return jsonify(response)
    elif task.state == "FAILURE":
        response = {
            "status": "FAILURE",
            "error": str(task.result),  # Make sure this is safe to display
        }
        return jsonify(response), 500
    else:
        return jsonify({"status": task.state})


@app.route("/check_url", methods=["POST"])
def check_url():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"message": "Please provide a URL."}), 400

    message = url_classifier.classify_url(url)

    return jsonify({"message": f"{message}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
