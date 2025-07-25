import requests


class CKAN:
    VALID_COUNTRIES = ["UK", "CANADA"]

    def __init__(self, country: str, base_url: str, action_url: str) -> None:
        assert country.upper() in self.VALID_COUNTRIES
        self._country = country
        self._base_url = base_url
        self._action_url = action_url
        self._final_url = f"{base_url}{action_url}"

    @staticmethod
    def show_valid_countries(cls):
        return cls.VALID_COUNTRIES

    def _make_request(self, url: str):
        """ "Do a GET request"""
        response = requests.get(url)

        return response.json()

    def _complete_url_with_kwargs(self, url, **kwargs):
        url += '&'.join(
            map(
                lambda x: f"{x[0]}={x[1]}",
                filter(
                    lambda v: v[1] is not None,
                    kwargs.items()
                )
            )
        )

        return url
        
    def package_search(self, **kwargs):
        match self._country:
            case "UK" | "CANADA":
                action = self._complete_url_with_kwargs("/package_search?", **kwargs)

        url = f"{self._final_url}{action}"
        
        return self._make_request(url)
    
    def package_show(self, **kwargs):
        match self._country:
            case "UK" | "CANADA":
                action = self._complete_url_with_kwargs("/package_show?", **kwargs)

        url = f"{self._final_url}{action}"
        return self._make_request(url)
        
    def resource_search(self, **kwargs):
        match self._country:
            case "UK" | "CANADA":
                action = self._complete_url_with_kwargs("/resource_search?", **kwargs)

        url = f"{self._final_url}{action}"
        return self._make_request(url)
    
    def resource_show(self, resource_id: str):
        match self._country:
            case "UK" | "CANADA":
                action = self._complete_url_with_kwargs("/resource_show?", id=resource_id)

        url = f"{self._final_url}{action}"
        return self._make_request(url)
    
    def download_tables_from_package_search(self, data_path: str, **package_search_kwargs):
        raise NotImplementedError()
