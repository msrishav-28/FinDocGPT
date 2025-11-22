import React, { useEffect, useState } from 'react'
import { API } from '../../services'
import { StatCard } from '../atoms'

export default function SentimentCard({ docId='demo_doc', ticker }){
  const [score, setScore] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(()=>{
    let ignore=false
    async function fetchSentiment(){
      setLoading(true); setError(null)
      try{
        const r = await API.get('/sentiment', { params: { doc_id: docId } })
        if(!ignore) setScore(r.data?.sentiment_score ?? 0)
      }catch(e){
        if(!ignore) setError('Failed to fetch sentiment')
      }finally{
        if(!ignore) setLoading(false)
      }
    }
    fetchSentiment()
    return ()=>{ ignore=true }
  }, [docId])

  if(loading) return <StatCard label="Sentiment" value="…" hint="Analyzing document" tone="gray"/>
  if(error) return <StatCard label="Sentiment" value="N/A" hint={error} tone="gray"/>
  const pct = ((Number(score)+1)/2*100).toFixed(0)
  const tone = Number(score) > 0.1 ? 'green' : Number(score) < -0.1 ? 'gray' : 'brand'
  return <StatCard label="Sentiment" value={`${pct}%`} hint="Normalized polarity" tone={tone} />
}
