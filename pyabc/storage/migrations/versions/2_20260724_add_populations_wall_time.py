"""add populations.wall_time and store global particle weights

Revision ID: 2
Revises: 1
Create Date: 2026-07-24 00:00:00.000000

This revision bundles two v2 storage-format changes:

* adds the ``populations.wall_time`` column (wall-time tracking), and
* converts particle weights from the old per-model normalization
  (``w = g_i / p_model``, summing to 1 within each model) to the global
  convention (``w = g_i``, summing to 1 across all particles of all models),
  matching the in-memory ``Population`` representation. The transform is
  ``w := w * p_model``; within-model normalization is recovered on read.
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
    # per-model-normalized weights -> global weights (w := w * p_model)
    op.execute(
        'UPDATE particles SET w = w * ('
        'SELECT p_model FROM models WHERE models.id = particles.model_id)'
    )


def downgrade():
    # global weights -> per-model-normalized weights (w := w / p_model)
    op.execute(
        'UPDATE particles SET w = w / ('
        'SELECT p_model FROM models WHERE models.id = particles.model_id)'
    )
