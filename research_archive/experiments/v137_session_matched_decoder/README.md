# V137 Session-Matched Paired Decoder

V137 isolates one previously unused piece of acquisition structure. Paired
source and target identities generally have multiple recording days, and each
day concentrates a different action script family. The existing decoder pools
all templates from a source user. V137 instead maps target days to source days
by relative chronological rank and offers only that day's templates.

The rule has no learned parameter and uses timestamps plus labeled source
scripts only. It never inspects target labels. V127 predictions remain exactly
unchanged for users without paired supervision.
