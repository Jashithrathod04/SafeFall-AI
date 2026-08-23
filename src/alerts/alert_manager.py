from datetime import datetime


class AlertManager:

    def __init__(self):

        self.active = False
        self.last_alert = None

    def trigger(self):

        self.active = True

        self.last_alert = (
            datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        return {
            "status": "EMERGENCY",
            "message":
                "Fall detected. "
                "Immediate assistance required.",
            "timestamp":
                self.last_alert,
        }

    def clear(self):

        self.active = False

    def get_status(self):

        return {
            "active": self.active,
            "last_alert":
                self.last_alert,
        }
