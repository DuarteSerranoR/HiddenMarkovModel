import time
import uvicorn
from fastapi import FastAPI, HTTPException

from hmm.hmm import HMM

model = HMM()
app = FastAPI()

from lib.logger import logger

#import logging
#logger = logging.getLogger("uvicorn.error")
#logger.propagate = False


@app.post("/pos_tag")
def pos_tag(txt: str):
    try:
        start_time = time.time()
        out = model.compute(txt)
        logger.info("ELAPSED {} ms".format(( (time.time() - start_time ) / 1000 )))
        return out
    except Exception as ex:
        logger.error(str(ex))
        return HTTPException(500, str(ex))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9091,
        #workers=
        #reload=True if DEBUG else False
        #log_level="debug"
    )
    logger.info("Ready to process requests!")
