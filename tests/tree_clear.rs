use lsm_tree::{get_tmp_folder, AbstractTree, Config, SeqNo, SequenceNumberCounter};
use test_log::test;

#[test]
fn tree_clear() -> lsm_tree::Result<()> {
    let folder = get_tmp_folder();

    let seqno = SequenceNumberCounter::default();
    let visible_seqno = lsm_tree::VisibleSeqno::default();

    let tree = Config::new(&folder, seqno.clone(), visible_seqno.clone()).open()?;

    assert_eq!(0, tree.len(visible_seqno.get(), None)?);

    {
        let seqno = seqno.next();
        tree.insert("a", "a", seqno);
        visible_seqno.begin(seqno).publish();
    }

    assert!(tree.contains_key("a", SeqNo::MAX)?);
    assert_eq!(1, tree.len(visible_seqno.get(), None)?);

    tree.clear()?;
    assert!(!tree.contains_key("a", SeqNo::MAX)?);
    assert_eq!(0, tree.len(visible_seqno.get(), None)?);

    {
        let seqno = seqno.next();
        tree.insert("a", "a", seqno);
        visible_seqno.begin(seqno).publish();
    }

    tree.flush_active_memtable(0)?;
    assert!(tree.contains_key("a", SeqNo::MAX)?);
    assert_eq!(1, tree.len(visible_seqno.get(), None)?);

    tree.clear()?;
    assert!(!tree.contains_key("a", SeqNo::MAX)?);
    assert_eq!(0, tree.len(visible_seqno.get(), None)?);

    Ok(())
}
