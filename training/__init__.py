import os
import sys

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import feed_forward_network
import graph_attention_network
import graph_convolutional_network
import recurrent_network
import our_network
