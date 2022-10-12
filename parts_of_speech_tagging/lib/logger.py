import logging

FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
logging.basicConfig(format=FORMAT,
                    level=logging.DEBUG)

#logger = logging.getLogger('hmm')
logger = logging.getLogger('uvicorn.error')

#logger.info("Done!")
