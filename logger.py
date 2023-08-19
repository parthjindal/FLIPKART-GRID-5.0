import logging
logging.basicConfig(
    level=logging.DEBUG, 
    format="[%(asctime)s:%(msecs)03d %(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",filename=f"./logs/logfile.log", filemode="w"
)