def logging_config_setup(logging_config):
    # %(name)s : uvicorn, uvicorn.error, ... . Not insightful at all.
    logging_config["formatters"]["access"][
        "fmt"
    ] = '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    logging_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s %(levelprefix)s %(message)s"

    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging_config["formatters"]["default"]["datefmt"] = date_fmt
    logging_config["formatters"]["access"]["datefmt"] = date_fmt
