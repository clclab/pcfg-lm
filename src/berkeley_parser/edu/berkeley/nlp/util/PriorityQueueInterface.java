package edu.berkeley.nlp.util;

public interface PriorityQueueInterface<E> {

	/**
	 * Returns true if the priority queue is non-empty
	 */
	public abstract boolean hasNext();

	/**
	 * Returns the element in the queue with highest priority, and pops it from
	 * the queue.
	 */
	public abstract E next();

	/**
	 * Not supported -- next() already removes the head of the queue.
	 */
	public abstract void remove();

	/**
	 * Returns the highest-priority element in the queue, but does not pop it.
	 */
	public abstract E peek();

	/**
	 * Gets the priority of the highest-priority element of the queue.
	 */
	public abstract double getPriority();

	/**
	 * Number of elements in the queue.
	 */
	public abstract int size();

	/**
	 * True if the queue is empty (size == 0).
	 */
	public abstract boolean isEmpty();

	/**
	 * Adds a key to the queue with the given priority. If the key is already in
	 * the queue, it will be added an additional time, NOT promoted/demoted.
	 * 
	 * @param key
	 * @param priority
	 */
	public abstract void put(E key, double priority);

}