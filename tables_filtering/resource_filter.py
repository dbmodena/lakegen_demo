from typing import List


def resource_filter(
    resources: List,
    format_filter: bool = True,
    language_filter: bool = True,
    resource_type_filter: bool = True,
):
    def format_f(resource):
        return not format_filter or (format_filter and resource["format"] == "CSV")

    def language_f(resource):
        return not language_filter or (
            language_filter and "language" in resource and "en" in resource["language"]
        )

    def resource_type_f(resource):
        return not resource_type_filter or (
            resource_type_filter
            and "resource_type" in resource
            and resource["resource_type"] == "dataset"
        )

    resources = list(
        filter(format_f, filter(language_f, filter(resource_type_f, resources)))
    )

    return resources
