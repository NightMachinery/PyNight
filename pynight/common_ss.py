##
#: * @tests
#: `semantic-scholar-get '--adder' 'FM' '--format' 'csv' '--flat' 'https://www.semanticscholar.org/paper/7ec5f207263100ea2d45db595712f611a74bafd9' 'https://www.semanticscholar.org/paper/Transformer-Interpretability-Beyond-Attention-Chefer-Gur/0acd7ff5817d29839b40197f7a4b600b7fba24e4'`
##
import datetime
import concurrent.futures
import argparse
import re
import requests
import json
from collections import OrderedDict
from pynight.common_json import JSONEncoderWithFallback
from pynight.common_debugging import ipdb_enable
from pynight.common_dict import simple_obj
from pynight.common_csv import dict_to_csv
from pynight.common_icecream import ic
import sys
from typing import (
    Iterable,
    List,
)


def semantic_scholar_paper_id_get(url):
    """
    Extract the Semantic Scholar paper ID from a URL.

    Args:
        url (str): The URL to extract the paper ID from.

    Returns:
        str or None: The extracted paper ID or None if not found.
    """

    # arXiv patterns
    arxiv_patterns = [
        r"(?i)/(?:abs|pdf)/(?:arxiv:)?([^/]+?)(?:\.pdf)?(?:#.*)?/*$",
        r"(?i)arxiv:([^/]+?)(?:\.pdf)?/*$",
        r"(?i)ar5iv.labs.arxiv.org/html/(\d+\.\d+)",
        r"(?i)semanticscholar.org/arxiv:([^/]+?)/*$",
        r"(?i)^https://scholar.google.com/.*&arxiv_id=([^/&]+)/*$",
        r"^https://(?:www\.)?doi\.org(?:.*)/arXiv\.([^/]+)",
        r".*/(\d+\.\d+)\.pdf$"
    ]

    # ACL patterns
    acl_patterns = [
        r"^https://(?:www\.)?aclanthology\.org/([^/]*)",
        r"^https://(?:www\.)?aclweb\.org/anthology/(?:.*/)?([^/]+)"
    ]

    # General Semantic Scholar patterns
    general_patterns = [
        r"^https://api.semanticscholar.org/([^?]+)$",
        r"^https://www.semanticscholar.org/paper/(?:(?:[^/]+)/)?([^/]{40})(?:/)?$"
    ]

    # Processing arXiv patterns
    for pattern in arxiv_patterns:
        match = re.search(pattern, url)
        if match:
            return "arxiv:" + match.group(1)

    # Processing ACL patterns
    for pattern in acl_patterns:
        match = re.search(pattern, url)
        if match:
            # Remove the '.pdf' suffix if it exists
            paper_id = re.sub(r'\.pdf$', '', match.group(1))
            return "ACL:" + paper_id

    # Processing general patterns
    for pattern in general_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def ss_api_get(paper_id):
    """
    Retrieve paper information from the Semantic Scholar API and return it as a dictionary.

    Args:
        paper_id (str): The paper identifier (e.g., arXiv:1705.10311 or Semantic Scholar ID).

    Returns:
        dict: A dictionary containing paper information.
    """
    # Base URL for the Semantic Scholar API
    base_url = "https://api.semanticscholar.org/graph/v1/paper/"

    # Construct the full URL with desired fields
    url = (
        f"{base_url}{paper_id}?fields=title,url,citationCount,influentialCitationCount,"
        "externalIds,abstract,venue,year,referenceCount,isOpenAccess,fieldsOfStudy,"
        "s2FieldsOfStudy,publicationTypes,publicationDate,journal,authors.name,"
        "authors.hIndex,authors.homepage,authors.affiliations,authors.citationCount,"
        "authors.paperCount,authors.aliases,authors.url,authors.externalIds,openAccessPdf"
    )

    try:
        # Send a GET request to the API
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            return data
        else:
            print(f"API request failed with status code {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def ss_augment(paper_info):
    ##
    paper_info['publicationDate'] = paper_info.get('publicationDate', None) or paper_info.get('year', '')
    ##
    current_date = datetime.datetime.now()
    year_month = current_date.strftime("%y %b")

    paper_info["date_added"] = year_month
    ##
    paper_id = paper_info["paperId"]

    connected_papers_url = f"https://www.connectedpapers.com/main/{paper_id}"
    paper_info["connectedPapersURL"] = connected_papers_url
    ##

    return paper_info


def ss_flatten(data):
    """
    Flatten a Semantic Scholar paper data dictionary.

    Args:
        data (dict): The Semantic Scholar paper data dictionary.

    Returns:
        dict: A flattened dictionary with appropriate keys.
    """
    flattened_data = dict(data)

    # Extract author information
    authors = data.get("authors", [])
    flattened_data["author_names"] = ", ".join(
        author.get("name", "") for author in authors if author.get("name", "")
    )
    # ic(flattened_data["author_names"])

    affiliations = [
        ", ".join(author.get("affiliations", ""))
        for author in authors
        if author.get("affiliations", "")
    ]
    # ic(affiliations)
    if affiliations:
        flattened_data["author_affiliations"] = ", ".join(affiliations)
        # ic(flattened_data["author_affiliations"])

    # Extract information for the first author
    if authors:
        first_author = authors[0]
        flattened_data["author_name"] = first_author.get("name", None)
        flattened_data["author_h_index"] = first_author.get("hIndex", None)
        flattened_data["author_homepage"] = first_author.get("homepage", None)
        flattened_data["author_citation_count"] = first_author.get(
            "citationCount", None
        )
        flattened_data["author_paper_count"] = first_author.get("paperCount", None)
        flattened_data["author_aliases"] = first_author.get("aliases", None)
        flattened_data["author_url"] = first_author.get("url", None)
        flattened_data["author_external_ids"] = first_author.get("externalIds", None)

    return vars(simple_obj(
        _drop_nones=True,
        **flattened_data,
    ))


def dict_flatten_json(data):
    def flatten_dict_helper(d, parent_key=""):
        items = []
        for key, value in d.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(flatten_dict_helper(value, new_key))
            elif isinstance(value, (str,)):
                items.append((new_key, value))
            elif isinstance(value, (List,)):
                items.append((new_key, ", ".join([str(v) for v in value])))
            else:
                items.append((new_key, value))
        return items

    flat_data = flatten_dict_helper(data)
    flattened_dict = dict(flat_data)
    return flattened_dict


class Result:
    def __init__(self, success, value=None, error_message=None):
        self.success = success
        self.value = value
        self.error_message = error_message

    def __repr__(self):
        return f"Result(success={self.success}, value={self.value}, error_message={self.error_message})"


def ss_get(
    urls: List[str],
    adder: str,
    flat_p: bool = False,
    mode: str = "v1",
    output_format: str = "json",
    json_indent: int = 2,
    parallel: bool = True,
) -> List[Result]:
    flat_p = flat_p or output_format == "csv"

    encoder = JSONEncoderWithFallback(
        indent=None if json_indent == "compact" else json_indent,
        fallback_function=str,
    )

    def process_url(url):
        try:
            paper_id = semantic_scholar_paper_id_get(url)

            if paper_id:
                paper_info = ss_api_get(paper_id)
                if paper_info:
                    paper_info = ss_augment(paper_info)
                    paper_info["adder"] = adder

                    if flat_p:
                        paper_info = ss_flatten(paper_info)

                        if output_format == "csv":
                            paper_info = dict_flatten_json(paper_info)

                    if mode == "v1":
                        paper_info["tags"] = ""
                        paper_info["priority"] = ""

                        paper_info_sel = dict()
                        key_mapping = OrderedDict(
                            [
                                ("adder", "Adder"),
                                ("tags", "Tags"),
                                ("title", "Title"),
                                ("priority", "Priority"),
                                ("date_added", "Date Added"),
                                ("citationCount", "Citations"),
                                # ("year", "Year"),
                                ("publicationDate", "Date Published"),
                                ("venue", "Venue"),
                                ("author_affiliations", "Affiliations"),
                                ("author_names", "Authors"),
                                ("url", "SemanticScholar"),
                                ("connectedPapersURL", "ConnectedPapers"),
                            ]
                        )

                        for k, v in paper_info.items():
                            if k in key_mapping:
                                k_new = key_mapping[k]
                                paper_info_sel[k_new] = v

                        paper_info_sel = OrderedDict(
                            (k, paper_info_sel.get(k, None))
                            for k in key_mapping.values()
                        )
                    elif mode == "all":
                        paper_info_sel = dict(paper_info)
                    else:
                        return Result(
                            success=False, error_message=f"Unsupported mode: {mode}"
                        )

                    if output_format == "json":
                        return Result(
                            success=True,
                            value=encoder.encode(dict(paper_info_sel)) + "\n",
                        )
                    elif output_format == "csv":
                        return Result(
                            success=True,
                            value=dict_to_csv(
                                paper_info_sel,
                                header_p=False,
                            ),
                        )
                    else:
                        return Result(
                            success=False,
                            error_message=f"Invalid output format specified for URL: {url}",
                        )

                else:
                    return Result(
                        success=False,
                        error_message=f"Failed to get the paper info from the API for URL: {url}",
                    )
            else:
                return Result(
                    success=False,
                    error_message=f"Paper ID extraction from URL failed for URL: {url}",
                )
        except Exception as e:
            return Result(success=False, error_message=str(e))

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:
            results = list(executor.map(process_url, urls))
    else:
        results = [process_url(url) for url in urls]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch paper information from Semantic Scholar API using a URL."
    )
    parser.add_argument(
        "urls", nargs="+", help="URLs of the papers on Semantic Scholar"
    )
    parser.add_argument(
        "--adder",
        default="",
        help="Specify the person adding this.",
    )
    parser.add_argument(
        "--parallel",
        type=bool,
        action=argparse.BooleanOptionalAction,
        # default=False,
        default=True,
        help="Enable parallel execution",
    )
    parser.add_argument(
        "--json_indent",
        default=2,
        help='Specify the JSON indentation amount or "compact".',
    )
    parser.add_argument(
        "--flat",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to print a flat dictionary",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (json or csv)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "v1"],
        default="v1",
        help="Output mode",
    )
    args = parser.parse_args()

    urls = args.urls
    mode = args.mode
    adder = args.adder
    output_format = args.format
    flat_p = args.flat
    json_indent = args.json_indent
    indent = None if json_indent == "compact" else int(json_indent)
    encoder = JSONEncoderWithFallback(indent=indent, fallback_function=str)
    parallel = args.parallel

    results = ss_get(
        urls,
        adder,
        flat_p=flat_p,
        mode=mode,
        output_format=output_format,
        json_indent=json_indent,
        parallel=parallel,
    )

    for result in results:
        if result.success:
            print(result.value, end="")
        else:
            print(result.error_message, file=sys.stderr)


if __name__ == "__main__":
    ipdb_enable()

    main()
