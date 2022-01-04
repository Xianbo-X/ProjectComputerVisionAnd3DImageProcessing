import rosbag
import rospy

def ExtractMessage(source_file:str,target_file:str,topic,num_msgs=5):
    """
    Extract num_message lines messages in the specific topic
    Reference: http://wiki.ros.org/ROS/Tutorials/reading%20msgs%20from%20a%20bag%20file

    Params:
    ======================
    source_file (str): full path to the bag file
    target_file (str): Save
    topic (str): Specific topic whose messages should be extracted
    num_msgs (int): number of messages to be extracted, default 5 messages
    Return:

    """

    assert (".bag" in source_file)
    bag_in=rosbag.Bag(source_file)
    total_count=0
    with open(target_file,"w") as fout:
        for topic,msg,t in bag_in.read_messages(topics=topic):
            fout.write("\n# =======================================")
            total_count+=1
            fout.write("\n")
            fout.write("# topic:           " + topic)
            fout.write("\n")
            fout.write("# msg_count:       %u\n" % total_count)
            fout.write("# timestamp (sec): {:.9f}\n".format(t.to_sec()))
            fout.write("# - - -\n")
            fout.write(str(msg))
            if total_count>=num_msgs:
                break
        fout.write("\n")
        fout.write("# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        fout.write("# Total messages found: {:>16}".format(total_count))
