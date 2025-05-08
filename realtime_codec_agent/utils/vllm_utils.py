import requests

def get_vllm_modelname(api_base, api_key="Empty", return_list=False):
    try:
        headers = {}
        if api_key != "Empty":
            headers = {"Authorization": "Bearer %s" % api_key}
        response = requests.get(f"{api_base}/models", headers=headers)
        if response.status_code == 200:
            # get the model name hosted by vllm
            models = [m for m in response.json()["data"] if m["object"] == "model"]
            if len(models) == 0:
                print("The vLLM server is running but not hosting any models.")
                return None
            if not return_list:
                model_name = models[0]["id"]
                print(f"The vLLM server is running and hosting model '{model_name}'.")
                return model_name
            else:
                model_names = [m["id"] for m in models]
                print(f"The vLLM server is running and hosting models: {model_names}.")
                return model_names
        else:
            print("The vLLM server is not running.")
            return None
    except requests.exceptions.ConnectionError:
        print("Could not connect to the server.")
        return None