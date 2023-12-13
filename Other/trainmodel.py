import configparser
import requests
import json

def main():
    config = configparser.ConfigParser(allow_no_value=True)
    config.read('./trainmodel.cfg')

    if 'url' not in config['main_url']:
        print("The configuration file must contain the valid Intent Detection Web Service's URL!")
    else:
        mainurl = config['main_url']['url']
        
        if 'json_training_data' in config and 'path' in config['json_training_data'] and 'name' in config['json_training_data'] and 'lang' in config['json_training_data']:
            print(f"Training model on {mainurl}")
            
            with open(config['json_training_data']['path'],'r',encoding='utf-8') as f:
                jsondict=json.load(f)
                
            txt = json.dumps(jsondict)
            
            payload = {'data': txt,
                'name':config['json_training_data']['name'],
                'lang':config['json_training_data']['lang'],
                'newmodel':'1'}

            response=requests.post(f"{mainurl}/train", data=payload, headers={'Accept':'text/json'})
            print(response.content)
        
        if  'remote_urls' in config:
            
            for singlekey in config['remote_urls'].keys():
                singleurl=config['remote_urls'][singlekey]
                print(f"Merging remote model from {singleurl}")
                
                payload = {'data': singleurl, 'newmodel':'0'}
                response=requests.post(f"{mainurl}/train", data=payload, headers={'Accept':'text/json'})
                print(response.content)
        
if __name__ == "__main__":
    main()
