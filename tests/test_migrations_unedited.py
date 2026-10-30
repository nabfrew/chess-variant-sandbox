"""Applied migrations are never edited, and the fast suite says so (Q1031).

Two deploys stopped at the migrate step because a mass rewrite touched one comment line in
an applied migration. The runner refuses on any change to the text, comments included. This
pins each file's digest, taken by the runner's own function, so the same edit fails here
and not on the box.
"""

from __future__ import annotations

from chess_variant_sandbox.site import migrate

# name -> sha256 of the file as the runner reads it. A NEW migration is added to this table
# in the same commit that adds the file. An existing row never changes.
PINNED = {
    "0001_store.sql": "43d616bc68911aca2f1d5f086d917e60c575a801089ed6d9e6686bfdfe6f8804",
    "0002_top_group.sql": "4e1254184458adc177449e15433f77f73f60c0e6fddaf850e7381c79a90ea43b",
    "0003_visitor_sandbox.sql": "2e90eadf1908575dcbebf588409240713b207910eaf6cac3f7a01decb7aa2a55",
    "0004_invite_email.sql": "1b7de2d9f92f345977b1d501cec44a8a97a2397a422b74e9204c9b6e59139c30",
    "0005_leaving.sql": "14dbb213706a43fc7be7b20eb3fed1921aa6e663ccc454514460faba687eff1f",
    "0006_usage.sql": "2b1c1d0e3e7c1a1859826768b41f53caf8a22d7ec902816ca70fb7408a489440",
    "0007_retired.sql": "1dfb791adf0cbc521b1f74b0dd5fc00d952f83a0c4528777dc36d4d23d8d76b2",
    "0008_awards.sql": "a7fecb9d26bdde44e06271724e24349de04b93cec7820bc4b2d4d63a9443275d",
    "0009_rematch.sql": "032bd63186c4c9360125c96d9826360bf5e8e63d6a01a3b4bfad5c7a3669120e",
    "0010_deleted.sql": "c121f61aa742efd41b35683a00933f97801085d0a2e4546ff21f74049b47bfbd",
    "0011_suggestions.sql": "ad23a1e15f1ef790b36284c890c449a07cc38e4e5a2c50f5ec2f48e911b6831d",
    "0012_links.sql": "fd2f1a3abadd0a60fce93e2333b2f0b93decddd579d74e79f0c5ad949e0707c9",
    "0013_similarity.sql": "eded529ae2e715e3fd8603978d5d300d1a6069717590084f35307f17d40157f3",
    "0014_depths.sql": "16f823dacb027f873fa3043b47ed0ae40cb02f2588548ef15adeac093d1ea204",
    "0015_provisional.sql": "85a6a34926c62d5b0a17b44837174baa5752563de080dba140474c31fd73a39f",
    "0016_drafts.sql": "4ca53e7a3b3f8d247b142b41dcad9414156ea967810cf641338acb69a230d04c",
    "0017_reports.sql": "e32d1a5f1fda69772b0429406950377b3ed9ab6fbae5bf85a42f5c5e1e595968",
    "0018_finishes.sql": "3612af2e0a2cd39f36c39f628f5c5abfb5df5bfe42378aabc129ab27f2558667",
    "0019_report_dismissed.sql": "2736deef3253470c09c98033cef36173b71e82233176518a350d4d6b132294d8",
    "0020_resting.sql": "c3fe5cf402bd3913dfef2d4de36cf18bc922ad43d6ca90a1e788fa7fc6fdf8df",
    "0021_rounds.sql": "0c8c083c945491e2f7b88eb1a10b4217360f7b20bd42506ba7763d73665dac4c",
    "0022_correspondence.sql": "1155e5f509663d946507a9fc967a276a74c5fb323b96634c81be61dcdff6716c",
    "0023_challenges.sql": "df5a2c7f83fa7d958f8d59455d66066576ba96c91c28c80aa2694e31273871a9",
    "0024_move_email.sql": "21a19473f7384f4e61bb2129890763eb6e4dec961ce254c33cdee19bebde043e",
    "0025_played.sql": "ec444cd6068edb69719b0779e55a5fa6b9b4d89d2a084549bacd244efc2a6b46",
    "0026_member_page.sql": "6dbbe78aa3d92955e5112e3f2c6599984578b3674c16369d4fad3af1f0e869da",
    "0027_takes_challengers.sql": "7fe5a7282d809678b280186f39149f75b97d8650b1a16776cfc96dc7d9ea20b1",
    "0028_invites_to_give.sql": "4b6ad1a51af95374bb59413d8ee3aebc836a9971fc8124cda848d12111f16e14",
    "0029_terms.sql": "858ca5d1a4f757386ff50e419aab0fee30be65623998a994800c4358bfba2f1a",
    "0030_spot_check.sql": "6de18b0ef23a0bc0c716272bc5aa995ff0eb01241645a0c786534b458743d58d",
    "0031_publish_waits.sql": "fd9758a0d0b99d3f9b8ef918407405d76db199f7e6c6f19b8d58cce28ae3cf08",
    "0032_human_data.sql": "0f5552554d7630b614e987b861f357da9cc24512ca78e57fba30dc084214a9e8",
    "0033_pinned_and_alarm.sql": "2d62dc1be307900e3df5aefc417c9f9d20eb92b42d8a6b18ca7b099992c5b2fe",
    "0034_invite_asks.sql": "66f828016470dbeaa24664c52b8212f631c09e72f2d7471b592b3a9d5f049ef8",
    "0035_draw_offers.sql": "69314535d3a0699e0fc1168ed48e95af4e1dfffd6d9daaeb23bdf695b0c4f79e",
    "0036_sealed_moves.sql": "0b96aad6690930fda7fc2d5f013f0c5c5de1a81a1e692c09ce2abc67d4e9ccb3",
    "0037_proven_positions.sql": "5d1f922735d393d04b4f3ca5b534fe59966930b9fff4710f6e777f41360e4c20",
    "0038_links_retired.sql": "7c545e79ca6a804d0965b78db426088a532d19ca4840757c82b9c8ef644c12f4",
    "0039_rest_on_pieces.sql": "7c4cc17091eebe9113155d0a9d571c170a9c97ab7d1dc3db5ad1a5587fcf8dc6",
    "0040_proven_names.sql": "43c1e7ad4c53baa144713dac89130d1023a7780c35844dc5d9d3177e50189fb8",
    "0041_one_spelling.sql": "041c74e56c1be0a22933add441cfb18f1354959141b47585463226cb55dcd88b",
    "0042_portal_moves.sql": "6e3369c6137e47d36ee7146ff08202d2892ad3be718bdab07d2080fe787cc84d",
    "0043_provisional_measured.sql": (
        "9c9a9835f07fcd85f9006864b7c38634268e10b2d1677a5dc11c61a72a52152b"),
}


def test_applied_migrations_are_unedited() -> None:
    actual = {m.name: m.checksum for m in migrate.available()}
    changed = sorted(n for n in PINNED if n in actual and actual[n] != PINNED[n])
    removed = sorted(set(PINNED) - set(actual))
    unpinned = sorted(set(actual) - set(PINNED))
    assert not changed and not removed, (
        f"applied migration(s) edited or removed: {changed + removed}. A migration is never "
        f"edited once applied, comments included (the box refuses and the site goes down). "
        f"Restore the file and put the change in a new numbered file."
    )
    assert not unpinned, (
        f"new migration(s) {unpinned} are not pinned: add each name and digest to PINNED in "
        f"the same commit that adds the file."
    )


def test_the_two_that_stopped_deploys_match_the_box() -> None:
    assert PINNED["0038_links_retired.sql"].startswith("7c545e79ca6a")
    assert PINNED["0040_proven_names.sql"].startswith("43c1e7ad4c53")
