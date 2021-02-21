"""add particles.proposal_id

Revision ID: 2
Revises:
Create Date: 2021-02-19 22:11:16.466274

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        table_name='particles',
        column=sa.Column(
            'proposal_id', sa.INTEGER, server_default='0'))


def downgrade():
    pass
