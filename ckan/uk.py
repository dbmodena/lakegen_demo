from .ckan import CKAN


class UKCKAN(CKAN):
    def __init__(self) -> None:
        super().__init__("UK", "https://data.gov.uk", "/api/action")
