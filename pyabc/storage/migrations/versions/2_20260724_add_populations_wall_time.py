"""add populations.wall_time

Revision ID: 2
Revises: 1
Create Date: 2026-07-24 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '2'
down_revision = '1'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        table_name='populations',
        column=sa.Column('wall_time', sa.FLOAT, nullable=True),
    )


def downgrade():
    pass
