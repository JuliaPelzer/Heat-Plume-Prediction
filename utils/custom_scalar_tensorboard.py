# + <
# We need these to make a custom protocol buffer to display custom scalars.
# See https://developers.google.com/protocol-buffers/
import tensorflow as tf
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.summary.v1 import custom_scalar_pb
#   > 
import numpy as np

def custom_scalar_tensorboard(train_loss:list, val_loss:list, name_folder: str = "default"):
    # Set up variables.
    init_values = np.array([0.0, 0.0])
    x = tf.Variable(init_values, name="train_loss", dtype=tf.float64)
    y = tf.Variable(init_values, name="val_loss", dtype=tf.float64)

    # List quantities to summarize in a dictionary 
    # with (key, value) = (name, Tensor).
    to_summarize = dict(
        train_loss = x,
        val_loss = y,
    )

    # Build scalar summaries corresponding to to_summarize.
    # This should be done in a separate name scope to avoid name collisions
    # between summaries and their respective tensors. The name scope also
    # gives a title to a group of scalars in TensorBoard.
    with tf.name_scope('loss_scalar_summary'):
        my_var_summary_op = tf.summary.merge(
            [tf.summary.scalar(name, var) 
                for name, var in to_summarize.items()
            ]
        )

    # + <
    # This constructs the layout for the custom scalar, and specifies
    # which scalars to plot.
    layout_summary = custom_scalar_pb(
        layout_pb2.Layout(category=[
            layout_pb2.Category(
                title='Custom scalar summary group',
                chart=[
                    layout_pb2.Chart(
                        title='Custom scalar summary chart',
                        multiline=layout_pb2.MultilineChartContent(
                            # regex to select only summaries which 
                            # are in "scalar_summaries" name scope:
                            tag=[r'^loss_scalar_summary\/']
                        )
                    )
                ])
        ])
    )
    #   >

    # Create session.
    with tf.Session() as sess:

        # Initialize session.
        sess.run(tf.global_variables_initializer())

        # Create writer.
        with tf.summary.FileWriter(f'runs/{name_folder}') as writer:

            # Write the session graph.
            writer.add_graph(sess.graph) # (not necessary for scalars)

    # + <
            # Define the layout for creating custom scalars in terms
            # of the scalars.
            writer.add_summary(layout_summary)
    #   >

            # Main iteration loop.
            for i in range(50):
                current_summary = sess.run(my_var_summary_op)
                writer.add_summary(current_summary, global_step=i)
                writer.flush()
