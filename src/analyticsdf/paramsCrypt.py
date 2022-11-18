import base64
import json

class paramsCrypt:

    def encode(parameters):
        # change dic to json (str)
        parametersStr = json.dumps(parameters)
        # encrypt json (str) to get sessionID (str)
        en = base64.b64encode(parametersStr.encode("utf-8"))
        return en
    
    def decode(enStr):
        # decrypt sessionID(str) to get original json(str)
        de = base64.b64decode(enStr).decode("utf-8")
        # retrieve dic from json (str)
        parametersStr = de.encode()
        parameters = json.loads(parametersStr.decode())
        return parameters

### Test
# test = { "Course Session":"ISE123-Analytics","Assignment Index":2,"Term":"Fall-2022", "column":[0.391, 984, 58.3] }
# enTest = paramsCrypt.encode(test)
# deTest = paramsCrypt.decode(enTest)
# print(deTest["column"])