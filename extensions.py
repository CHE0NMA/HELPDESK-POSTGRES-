"""Shared Flask extension instances, created here (not inside app.py) so
blueprint files can import `csrf` and call csrf.exempt() on individual
routes without a circular import back to app.py."""
from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect()