from base_plugin import BasePlugin
from ui.settings import Header, Text, Switch, Divider
from android_utils import run_on_ui_thread, log
from client_utils import run_on_queue
from java import dynamic_proxy
from java.util.function import Consumer
