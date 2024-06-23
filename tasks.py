# tasks.py
from celery_app import celery
from celery.utils.log import get_task_logger
from report_generation import generate_detailed_response_report, comment_each_section
from querytweet import country_stats, industry_stats
from queryNCSI import ncsi_insights
from queryITU import gci_insights
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging

# Set up a logger specifically for Celery tasks
logger = get_task_logger(__name__)

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)


@celery.task
def generate_report_async(
    response_reports,
    category_averages,
    country,
    industry,
    overall_score_message,
    detailed_results,
):
    logger.info(f"Starting task for country: {country} and industry: {industry}")
    try:
        gci_standing = ncsi_standing = country_situation = section_comments = None

        if country != "Hong Kong":
            # Log the start of parallel processing
            logger.info("Starting parallel processing for GCI, NCSI, and country stats")
            with ThreadPoolExecutor() as executor:
                future_to_function = {
                    executor.submit(gci_insights, country): "gci_standing",
                    executor.submit(ncsi_insights, country): "ncsi_standing",
                    executor.submit(country_stats, country): "country_situation",
                }
                logger.info(f"Number of threads used: {executor._max_workers}")

                results = {}
                for future in as_completed(future_to_function):
                    function_name = future_to_function[future]
                    try:
                        result = future.result()
                        results[function_name] = result
                    except Exception as e:
                        logger.error(
                            f"Error running {function_name}: {e}", exc_info=True
                        )
                        results[function_name] = None

                gci_standing = results.get("gci_standing")
                ncsi_standing = results.get("ncsi_standing")
                country_situation = results.get("country_situation")

        if category_averages is not None and detailed_results is not None:
            section_comments = comment_each_section(
                response_reports,
                category_averages,
                overall_score_message,
                detailed_results,
            )

        industry_situation = industry_stats(industry)

        logger.info("Task completed successfully")
        return {
            "gci_standing": gci_standing,
            "ncsi_standing": ncsi_standing,
            "section_comments": section_comments,
            "country_situation": country_situation,
            "industry_situation": industry_situation,
        }
    except Exception as e:
        logger.error(f"Error in generate_report_async: {e}", exc_info=True)
        raise
