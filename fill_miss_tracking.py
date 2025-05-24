def fill_results(tracking_results, detection_results):
    if len(tracking_results)!=len(detection_results):
        print(f"warning, tracking results is {len(tracking_results)}, while detection is {len(detection_results)}")
    for i, (frame, ball_tracking, player_tracking) in enumerate(tracking_results):
        if not(ball_tracking):
            print("oops")

            print(type(detection_results[i][1]))
            print(detection_results[i][1])
        else:
            print("tracking:")
            print(type(player_tracking))

