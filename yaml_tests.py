import os
from jinja2 import Environment, FileSystemLoader
import yaml

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
print(os.path.realpath(__file__), DIR_PATH)
env = Environment(loader=FileSystemLoader(DIR_PATH))
template = env.get_template("config/train_config.yaml")
c = template.render(n_res=10, val=10)
# yaml :)
config_vals = yaml.safe_load(c)
asd = {"qwe": 123}
asd.update(config_vals)
print(asd)
print(asd['nb_col'])
