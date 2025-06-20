FROM ros:noetic

# Expose ports used by rospy.init_node() in hardware.py
EXPOSE 45100
EXPOSE 45101

# Safely remove old ROS key sources if they exist
RUN rm -f /etc/apt/sources.list.d/ros1-latest.list \
    && rm -f /usr/share/keyrings/ros1-latest-archive-keyring.gpg

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    ros-noetic-opencv-apps \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Add updated ros-apt-source
RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -s -o /tmp/ros-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" && \
    apt-get update && \
    apt-get install -y /tmp/ros-apt-source.deb && \
    rm -f /tmp/ros-apt-source.deb

# Install ROS package example
RUN apt-get update && apt-get install -y ros-noetic-roscpp-tutorials && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY ./requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && pip install -r /requirements.txt && rm /requirements.txt

# Set working directory and copy catkin workspace
WORKDIR /root/catkin_ws
COPY ./catkin_ws ./

# Copy entry/setup scripts
COPY ./scripts/entrypoint.bash ./entrypoint.bash
COPY ./scripts/setup.bash ./setup.bash

# Convert line endings (Windows users) and clean up
RUN find . -type f \( -name '*.py' -o -name '*.bash' \) -exec dos2unix -l -- '{}' \; && \
    apt-get purge -y dos2unix && rm -rf /var/lib/apt/lists/*

# Compile the catkin workspace
RUN bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Make scripts executable
RUN chmod +x ./entrypoint.bash && chmod +x ./setup.bash && chmod -R u+x /root/catkin_ws/

# Set entrypoint
ENTRYPOINT ["./entrypoint.bash"]
